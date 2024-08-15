import os
import csv
from pydub import AudioSegment
from pydub.silence import detect_silence
import argparse
import logging
import whisper

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_silences(audio, min_silence_len=500, silence_thresh=-40):
    """
    Detect significant silences in the audio.
    """
    logging.info("Detecting silences in the audio...")
    silences = detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    print(silences)
    return [(start, end) for start, end in silences]

def split_audio_by_silences(audio, silences):
    """
    Split the audio based on the identified silence points.
    """
    logging.info("Splitting audio at identified silence points...")
    chunks = []
    prev_end = 0

    for start, end in silences:
        chunk = audio[prev_end:((end - start)/2 + start)]
        if chunk.duration_seconds > 0:
            chunks.append(chunk)
        prev_end = end
    
    # Add the last chunk
    if prev_end < len(audio):
        chunks.append(audio[prev_end:])

    logging.info(f"Created {len(chunks)} initial chunks based on silence.")
    return chunks

def merge_short_chunks(chunks, min_duration, max_duration):
    """
    Merge chunks that are shorter than the minimum duration with adjacent chunks
    so that the resulting chunks are between min_duration and max_duration.
    
    Parameters:
    - chunks: List of AudioSegment objects that have been split by silence.
    - min_duration: The minimum duration (in milliseconds) that a merged chunk should have.
    - max_duration: The maximum duration (in milliseconds) that a merged chunk should have.
    
    Returns:
    - A list of AudioSegment objects where each segment is within the specified duration constraints.
    """

    logging.info("Merging chunks to meet duration constraints...")

    # Initialize an empty list to hold the final merged chunks
    merged_chunks = []

    # Temporary chunk used for merging smaller chunks
    temp_chunk = AudioSegment.empty()

    for chunk in chunks:
        # Check if adding the current chunk to the temp_chunk would keep it under the min_duration
        if len(temp_chunk) + len(chunk) < min_duration:
            temp_chunk += chunk  # Merge the chunk with the temp_chunk
        else:
            # If the temp_chunk meets the minimum duration, check its length against max_duration
            if len(temp_chunk) >= min_duration:
                if len(temp_chunk) <= max_duration:
                    # If the temp_chunk is within the acceptable range, add it to the final list
                    merged_chunks.append(temp_chunk)
                else:
                    # If temp_chunk exceeds max_duration, handle the overflow
                    logging.warning(f"Chunk exceeded max_duration: {len(temp_chunk) / 1000:.2f} seconds")
                    merged_chunks.append(temp_chunk[:max_duration])  # Add the first part
                    temp_chunk = temp_chunk[max_duration:] + chunk  # Keep the overflow part and add the current chunk
            else:
                # If temp_chunk doesn't meet min_duration, continue adding to it
                temp_chunk += chunk
            
            # If temp_chunk now exceeds max_duration after adding the current chunk
            if len(temp_chunk) > max_duration:
                merged_chunks.append(temp_chunk[:max_duration])  # Add up to max_duration
                temp_chunk = temp_chunk[max_duration:]  # Keep the overflow part
            else:
                merged_chunks.append(temp_chunk)  # Otherwise, add the complete temp_chunk
                temp_chunk = AudioSegment.empty()  # Reset temp_chunk for the next iteration

    # After the loop, handle any remaining temp_chunk
    if len(temp_chunk) >= min_duration:
        if len(temp_chunk) <= max_duration:
            merged_chunks.append(temp_chunk)  # Add if it meets the criteria
        else:
            # If the final temp_chunk exceeds max_duration, split and log
            logging.warning(f"Final chunk exceeded max_duration: {len(temp_chunk) / 1000:.2f} seconds")
            merged_chunks.append(temp_chunk[:max_duration])
    else:
        # Log a warning if the last chunk doesn't meet min_duration
        logging.warning(f"Final chunk did not meet min_duration: {len(temp_chunk) / 1000:.2f} seconds")
    
    logging.info(f"Merged down to {len(merged_chunks)} chunks after applying duration constraints.")
    return merged_chunks

def export_chunks_and_generate_metadata(model, chunks, output_dir):
    """
    Export audio chunks and generate corresponding metadata.
    """
    metadata = []
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Created output directory: {output_dir}")

    for i, chunk in enumerate(chunks):
        segment_filename = f"{str(i + 1).zfill(2)}.wav"
        segment_filepath = os.path.join(output_dir, segment_filename)
        
        logging.info(f"Exporting segment to file: {segment_filepath}")
        chunk.export(segment_filepath, format="wav")
        logging.info(f"Segment exported. Duration: {len(chunk) / 1000:.2f} seconds")
        
        # Metadata could be added here (e.g., filename, transcription, etc.)
        result = model.transcribe(segment_filepath)
        metadata.append((segment_filename[:-4], "speaker", result["text"]))

    return metadata

def save_metadata_files(output_dir, metadata):
    """
    Save metadata to CSV, train, and validation files.
    """
    metadata_file = os.path.join(output_dir, 'metadata.csv')
    train_file = os.path.join(output_dir, 'train.txt')
    valid_file = os.path.join(output_dir, 'valid.txt')

    logging.info("Saving metadata to CSV file...")
    with open(metadata_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='|')
        writer.writerows(metadata)
    
    logging.info("Splitting metadata into training and validation sets...")
    num_entries = len(metadata)
    split_point = int(num_entries * 0.9)  # 90% for training, 10% for validation
    train_data = metadata[:split_point]
    valid_data = metadata[split_point:]

    logging.info("Saving training data to train.txt...")
    with open(train_file, 'w') as trainfile:
        for entry in train_data:
            trainfile.write('|'.join(entry) + '\n')

    logging.info("Saving validation data to valid.txt...")
    with open(valid_file, 'w') as validfile:
        for entry in valid_data:
            validfile.write('|'.join(entry) + '\n')

def main():
    """
    Main function to process the audio file and generate LJSpeech dataset format.
    """
    parser = argparse.ArgumentParser(description="Process an audio file into LJSpeech dataset format.")
    parser.add_argument('audio_file', type=str, help='Path to the audio file')
    parser.add_argument('--min_duration', type=int, default=6000, help='Minimum duration for segments in milliseconds')
    parser.add_argument('--max_duration', type=int, default=11600, help='Maximum duration for segments in milliseconds')
    parser.add_argument('--min_silence_len', type=int, default=500, help='Minimum length of silence to consider for splitting (in ms)')
    parser.add_argument('--silence_thresh', type=int, default=-40, help='Silence threshold (in dB)')
    parser.add_argument('--model_size', type=str, default='base', help='Size of the Whisper model to use (e.g., base, small, medium, large)')
    args = parser.parse_args()

    audio_file = args.audio_file
    min_duration = args.min_duration
    max_duration = args.max_duration
    min_silence_len = args.min_silence_len
    silence_thresh = args.silence_thresh
    model_size = args.model_size
    output_dir = os.path.join(os.path.dirname(audio_file), 'wavs')
    
    logging.info(f"Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)
    
    logging.info(f"Starting processing for file: {audio_file}")
    
    # Load the audio file
    audio = AudioSegment.from_file(audio_file)
    
    # Find all significant silences in the audio
    silences = find_silences(audio, min_silence_len, silence_thresh)
    
    # Split the audio into initial chunks based on the identified silences
    initial_chunks = split_audio_by_silences(audio, silences)

    final_chunks = merge_short_chunks(initial_chunks, min_duration, max_duration)
    
    # Export the chunks and generate metadata
    metadata = export_chunks_and_generate_metadata(model, final_chunks, output_dir)
    
    # Save the metadata to files
    save_metadata_files(os.path.dirname(audio_file), metadata)

    logging.info("Processing complete.")

if __name__ == "__main__":
    main()
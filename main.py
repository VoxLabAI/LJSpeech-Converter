import os
import argparse
import whisper
import csv


def transcribe_audio(file_path: str):
    """
    Converts the audio into text

    :param file_path: Name of the wav file to transcript
    :return: Transcription of the wav file
    """
    model = whisper.load_model("base")  # or use "small", "medium", "large", depending on your needs and resources
    result = model.transcribe(file_path)
    return result['text']


def main(dataset: str):
    """
    Create metadata in LJSpeech dataset format based on the wav files

    :param dataset: folder of all the wav files to transcript
    """
    metadata = []

    # List all WAV files in the specified folder
    for filename in os.listdir(dataset):
        if filename.endswith(".wav"):
            file_path = os.path.join(dataset, filename)
            print(f"Processing {file_path}...")

            # Transcribe audio and create metadata entry
            try:
                transcription = transcribe_audio(file_path)
                metadata.append([filename[:-4], transcription])  # Remove .wav extension from filename
            except Exception as e:
                print(f"Failed to transcribe {filename}: {e}")

    # Write metadata.csv
    metadata_path = os.path.join(dataset, "metadata.csv")
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerows(metadata)
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LJSpeech-like dataset using OpenAI Whisper.")
    parser.add_argument("--dataset", type=str, help="Path to the folder containing WAV files")
    args = parser.parse_args()
    main(**vars(args))

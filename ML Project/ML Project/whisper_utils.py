import whisper
import os

class WhisperModel:
    def __init__(self):

        self.model = whisper.load_model("small")


    def transcribe(self, audio_file):
        try:
            if not os.path.exists(audio_file):
                print(f"Audio file {audio_file} does not exist.")
                return None

            result = self.model.transcribe(audio_file)
            print("Transcription complete.")
            return result["text"]
        except Exception as e:
            print(f"An error occurred during transcription: {e}")
            return None

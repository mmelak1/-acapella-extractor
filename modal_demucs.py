import io
import modal


def _download_model():
    from demucs.pretrained import get_model
    get_model("mdx_q")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "demucs",
        "diffq",
        "torch==2.3.0",
        "torchaudio==2.3.0",
        "soundfile",
    )
    .run_function(_download_model)
)

app = modal.App("acapella-extractor", image=image)


@app.cls(gpu="A10G", timeout=300, scaledown_window=600)
class VocalSeparator:
    @modal.enter()
    def load(self):
        from demucs.pretrained import get_model
        self.model = get_model("mdx_q")
        self.model.cuda()
        self.model.eval()
        self.vocal_idx = list(self.model.sources).index("vocals")

    @modal.method()
    def separate_vocals(self, audio_bytes: bytes, filename: str) -> dict:
        import subprocess
        import torch
        import torchaudio
        from demucs.apply import apply_model

        wav, sr = torchaudio.load(io.BytesIO(audio_bytes))

        if sr != self.model.samplerate:
            wav = torchaudio.functional.resample(wav, sr, self.model.samplerate)

        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        wav = wav[:2]

        with torch.no_grad():
            sources = apply_model(self.model, wav.unsqueeze(0).cuda(), shifts=0, overlap=0.1)

        vocals = sources[0, self.vocal_idx].cpu()

        wav_buf = io.BytesIO()
        torchaudio.save(wav_buf, vocals, self.model.samplerate, format="wav")
        wav_bytes = wav_buf.getvalue()

        mp3 = subprocess.run(
            ["ffmpeg", "-i", "pipe:0", "-q:a", "2", "-f", "mp3", "pipe:1"],
            input=wav_bytes, capture_output=True,
        )

        return {"mp3": mp3.stdout if mp3.returncode == 0 else wav_bytes}

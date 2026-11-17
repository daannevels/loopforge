import numpy as np
import pytest
import soundfile as sf

from loopforge.audio_io import file_sha1, is_audio_file, load_audio, resample, to_mono, to_stereo, write_wav
from tests.synth import SR, basic_pattern, render_loop, write


@pytest.mark.parametrize("ext", [".wav", ".aiff", ".flac"])
def test_roundtrip_formats(tmp_path, ext):
    x = render_loop(basic_pattern(), 128.0, 1)
    p = write(tmp_path / f"loop{ext}", x)
    assert is_audio_file(p)
    y, sr = load_audio(p)
    assert sr == SR
    assert y.shape == x.shape and y.dtype == np.float32
    assert np.max(np.abs(y - x)) < 1e-4  # 24-bit quantisation


def test_mono_loads_as_one_channel_and_writes_stereo_24bit(tmp_path):
    x = render_loop(basic_pattern(), 128.0, 1, stereo=False)
    p = write(tmp_path / "mono.wav", x)
    y, _ = load_audio(p)
    assert y.shape[0] == 1
    out = write_wav(tmp_path / "out.wav", y, SR)
    info = sf.info(str(out))
    assert info.channels == 2 and info.subtype == "PCM_24" and info.samplerate == SR
    z, _ = load_audio(out)
    assert np.allclose(z[0], z[1])


def test_stereo_mono_helpers():
    m = np.ones(10, np.float32)
    assert to_stereo(m).shape == (2, 10)
    assert to_mono(np.stack([m, -m])).max() == 0.0


def test_resample_length_and_tone():
    t = np.arange(44100) / 44100
    x = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    y = resample(x, 44100, 48000)
    assert len(y) == 48000
    spec = np.abs(np.fft.rfft(y))
    assert abs(np.argmax(spec) * 48000 / len(y) - 1000) < 2


def test_sha1_changes_with_content(tmp_path):
    a = write(tmp_path / "a.wav", render_loop(basic_pattern(), 120.0, 1))
    b = write(tmp_path / "b.wav", render_loop(basic_pattern(), 121.0, 1))
    assert file_sha1(a) != file_sha1(b)
    assert len(file_sha1(a)) == 40

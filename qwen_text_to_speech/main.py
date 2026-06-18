import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# Load the model
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map="mps",
    dtype=torch.bfloat16,
)

ref_audio = "The Quick Brown Fox.wav"
ref_text = "The quick brown fox jumped over the lazy dog."

print("Generating speech...")
# Generate speech
wavs, sr = model.generate_voice_clone(
    text="""Goodbye Ellie!
    Have a nice day at kindy!
    Hope you have fun building houses with Jessen!
    Love you!

    """,
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
)

print("Saving audio...")
# Save the resulting audio
sf.write("ellie4.wav", wavs[0], sr)

"""Piranhas don't eat bananas, by Aaron Blabey.
    Hey there, guys. Would you like a banana?
    What's wrong with you, Brian?
    You're a piranha.

    'Well, how about some silverbeet?'
    'Are you serious, Brian? We eat feet.'

    'Or would you rather a bowl of peas?'
    'Stop it, Brian. We eat knees.'

    'We, I bet you'd like some juicy plums?'
    'That's it, Brian! We eat bums!'
    'We don't eat apples! 
    we don't eat beans! 
    we don't eat veggies! 
    We don't eat greens! 
    We don't eat melons! 
    We don't eat bananas!
    And the reason is simple, mate. We are piranhas.'

    """
# Image to Ambient Audio

This project is a simple, curiosity-driven experiment to generate ambient audio from an image. It includes a basic Python script (`image2sound.py`) that analyzes an image and creates a soundscape that reflects the image's content.

## How It Works

The script uses a multi-step process to generate audio from an image:

1.  **Image Captioning**: A model like BLIP is used to describe the contents of the photo in text (e.g., "a beautiful beach with waves crashing on the shore").
2.  **Prompt Engineering**: A descriptive audio prompt is constructed from the generated caption, focusing on ambient sounds.
3.  **Audio Generation**: An audio generation model (like Stable Audio) synthesizes a stereo WAV file based on the text prompt.

## Use Cases

This technology can be applied in various creative and practical ways:

*   **Meditation & Relaxation Apps**: Automatically generate calming soundscapes from serene images like forests, beaches, or mountains to aid in meditation and sleep.
*   **AR/VR & Gaming**: Enhance immersive experiences by dynamically creating ambient audio that matches the visual environment of a game or virtual world.
*   **Content Creation**: Quickly generate background audio for videos, podcasts, or presentations based on a theme or image.
*   **Prototyping**: A valuable tool for quickly prototyping sound design for films, games, and interactive installations.

## Examples

*(Tip: GitHub mutes videos by default — don’t forget to turn the sound on in the player to hear the generated ambience.)*

<div align="center">

<h3>Beach Scene</h3>
<video
  src="https://github.com/user-attachments/assets/b640d31e-b92f-4eab-8249-7b3cb434b757"
  controls
  muted
  playsinline
  width="512"
  poster="examples/beach.jpeg">
  Sorry, your browser doesn’t support embedded videos.
</video>
<br/>
<small>(Generation peaked at ~9.2 GB VRAM.)</small>

<br/><br/>

<h3>Office Environment</h3>
<video
  src="https://github.com/user-attachments/assets/d99376af-bd14-4cd7-b6d1-772391e11ffc"
  controls
  muted
  playsinline
  width="512"
  poster="examples/office.jpg">
</video>

</div>

---

This repository contains a basic script to explore this concept and was primarily created out of curiosity.

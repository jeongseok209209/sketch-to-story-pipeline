Put ordered story images here, or group each story in its own folder.

Flat input still works:

- 01.png
- 02.png
- 03.png

Story-folder input also works:

- 1. The Rabbit and the Turtle/
  - 1.png
  - 2.png
  - 3.png
- 2. The Fox and the Crane/
  - 1.png
  - 2.png
  - 3.png

When story folders exist, `python run.py g` asks which folder to use. You can skip the prompt with `--story`, for example:

```powershell
python run.py g --story 1
python run.py g --story "1. The Rabbit and the Turtle"
```

To run all experiments for a story and immediately open blind evaluation from successful results only:

```powershell
python run.py all-evaluate --story 1 --port 8501
```

The sequence mode reads PNG/JPG/JPEG files in numeric filename order and creates one connected story.

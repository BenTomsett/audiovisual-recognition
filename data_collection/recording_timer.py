import random
import threading
import time
import librosa
import tkinter as tk
from tkinter import ttk
import sounddevice as sd


def play_tone():
    """
    Play a 440 Hz tone for 1 second.
    """
    tone = librosa.tone(440, sr=44100, length=int(44100))
    sd.play(tone, 44100)
    sd.wait()


def read_names():
    """
    Reads in a list of names, repeats them 20 times, and shuffles them.
    :return: array of names
    """
    with open("names.txt", "r") as file:
        _names = file.read().splitlines()
        _names *= 20
        random.shuffle(_names)
        return _names


def update_progress_bar():
    """
    Runs a progress bar for 1 second.
    """
    for i in range(101):
        time.sleep(0.01)
        progress_var.set(i)
    window.after(0, next_name)


def next_name():
    """
    Updates the name label to the next name.
    """
    name_label.config(fg="red")
    window.after(5, start_display)


def start_display():
    """
    This function is repeatedly called to update the display with the next name.
    """
    global name_index

    play_tone()

    if name_index < len(names):
        name_label.config(text=names[name_index], fg="black")
        name_index += 1
        count_label.config(text="Recording {} of {}".format(name_index, len(names)))
        threading.Thread(target=update_progress_bar).start()
    else:
        name_label.config(text="Finished.", fg="green")


names = read_names()
name_index = 0

window = tk.Tk()
window.geometry("500x250")
window.title("Recording Timer")

count_label = tk.Label(window, text="--", font=("SF Mono", 20))
count_label.pack(pady=10)

name_label = tk.Label(window, text="Ready to start", font=("SF Mono", 50), fg="green")
name_label.pack(pady=20)

progress_var = tk.IntVar()
progress_bar = ttk.Progressbar(window, length=450, variable=progress_var)
progress_bar.pack(pady=10)

start_button = tk.Button(
    window,
    text="Start",
    command=start_display,
)
start_button.pack(pady=10)

window.mainloop()

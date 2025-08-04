# Made by chatGPT op 13-03-2025 als workaround om LyX tekst te kunnen spellchecken met hunspell
# Dit omdat de spellchecker in Lyx niet werkt.
# @TO 20250313

# %%
import os
import subprocess

# Paths
lyx_app = "/System/Volumes/Data/Applications/LyX.app/Contents/MacOS/lyx"
doc_path = "/Users/Theo/GRWMODELS/python/Stromingen/H2O051995/lyx/"


# Files
fname = "Maas1995_dik_freatisch_pakket_infiltreert_niet_lekker.lyx"
lyx_file = os.path.join(doc_path, fname)
tex_file = lyx_file.replace(".lyx", ".tex")

# Step 1: Export .lyx to .tex using LyX if the .tex doesn't exist
if not os.path.exists(tex_file):
    cmd = [lyx_app, "--export", "pdflatex", str(lyx_file)]
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully exported {lyx_file} to {tex_file}.")
    except subprocess.CalledProcessError as e:
        print(f"Error during LyX export: {e}")
        exit(1)

# Step 2: Run aspell on the .tex file
aspell_cmd = f"aspell -d nl list --mode=tex < {tex_file} | sort -u"

try:
    misspelled_words = subprocess.check_output(aspell_cmd, shell=True, text=True)
    print("Misspelled words:")
    print(misspelled_words)
except subprocess.CalledProcessError as e:
    print(f"Error during aspell check: {e}")
    exit(1)

# %%

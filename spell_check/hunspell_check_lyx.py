import os
import subprocess

# Paths
lyx_app = "/System/Volumes/Data/Applications/LyX.app/Contents/MacOS/lyx"
doc_path = "/Users/Theo/Entiteiten/Hygea/2022-AGT/Doc_rapportage/"
os.chdir(doc_path)

# Files
fname = "VrijeAfwatering.lyx"
lyx_file = os.path.join(doc_path, fname)
tex_file = lyx_file.replace(".lyx", ".tex")
output_file = "misspelled_words.txt"

# Step 1: Export .lyx to .tex using LyX if needed
if not os.path.exists(tex_file):
    cmd = [lyx_app, "--export", "pdflatex", lyx_file]
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Exported {lyx_file} to {tex_file}.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during LyX export: {e}")
        exit(1)

# Step 2: Run delatex to remove LaTeX
try:
    filtered_tex = subprocess.check_output(["delatex", tex_file], text=True)
    print("✅ Removed LaTeX markup with delatex.")
except subprocess.CalledProcessError as e:
    print(f"❌ Error during delatex: {e}")
    exit(1)

# Step 3: Run hunspell and capture output
try:
    hunspell = subprocess.Popen(
        ["hunspell", "-d", "nl_NL", "-l"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    output, _ = hunspell.communicate(filtered_tex)
    misspelled = sorted(set(output.strip().splitlines()))
    print("✅ Misspelled words:")
    print("\n".join(misspelled))

    # Step 4: Write results to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(misspelled))
    print(f"📝 Misspelled words saved to '{output_file}'")

except subprocess.CalledProcessError as e:
    print(f"❌ Error during hunspell: {e}")
    exit(1)

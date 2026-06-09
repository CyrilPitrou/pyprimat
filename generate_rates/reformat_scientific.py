import sys

if len(sys.argv) < 2 or len(sys.argv) > 3:
    print("Usage: python reformat_scientific.py <input_file> [output_file]")
    sys.exit(1)

input_file  = sys.argv[1]
output_file = sys.argv[2] if len(sys.argv) == 3 else input_file

with open(input_file, "r") as fin:
    lines = fin.readlines()

output_lines = []
for line in lines:
    # Keep comment lines untouched
    if line.startswith("#"):
        output_lines.append(line)
        continue

    # Reformat each number in the line
    parts = line.split()
    if parts:
        nums = [float(x) for x in parts]
        formatted = f"{nums[0]:.6E}" + "".join(f"{x:>20.6E}" for x in nums[1:])
        output_lines.append(formatted + "\n")

with open(output_file, "w") as fout:
    fout.writelines(output_lines)

print(f"Done. Output written to '{output_file}'")

import os

def create_svg(letter):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <rect width="200" height="200" fill="#374151" rx="20"/>
  <text x="50%" y="45%" dominant-baseline="middle" text-anchor="middle" font-family="sans-serif" font-weight="bold" font-size="90" fill="#A8E6CF">{letter}</text>
  <text x="50%" y="80%" dominant-baseline="middle" text-anchor="middle" font-family="sans-serif" font-weight="600" font-size="20" fill="white">ASL Sign</text>
</svg>"""

output_dir = r"d:\mini project\SIGNARA-main\frontend\public\gestures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

missing_letters = [chr(i) for i in range(ord('O'), ord('Z')+1)]
for letter in missing_letters:
    path = os.path.join(output_dir, f"{letter}.svg")
    with open(path, "w") as f:
        f.write(create_svg(letter))

print("Created SVG placeholders for O-Z.")

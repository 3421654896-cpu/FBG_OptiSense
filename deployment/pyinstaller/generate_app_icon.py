"""Generate the Windows application icon from the program's own logo."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def main() -> int:
    output = Path(__file__).with_name("fbg_optisense_studio.ico")
    size = 256
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle((5, 5, 250, 250), radius=68, fill=255)
    gradient = Image.new("RGBA", (size, size))
    pixels = gradient.load()
    for y in range(size):
        for x in range(size):
            position = (x + y) / (2 * (size - 1))
            if position <= 0.52:
                fraction = position / 0.52
                first, second = (102, 181, 255), (52, 130, 255)
            else:
                fraction = (position - 0.52) / 0.48
                first, second = (52, 130, 255), (103, 92, 255)
            pixels[x, y] = tuple(
                round(first[index] + (second[index] - first[index]) * fraction)
                for index in range(3)
            ) + (255,)
    image.paste(gradient, (0, 0), mask)
    draw = ImageDraw.Draw(image)
    wave = [(45, 146), (67, 89), (93, 187), (126, 110), (157, 64), (187, 178), (213, 106)]
    draw.line(wave, fill="white", width=16, joint="curve")
    for point in wave:
        draw.ellipse((point[0] - 8, point[1] - 8, point[0] + 8, point[1] + 8), fill="white")
    image.save(
        output,
        format="ICO",
        sizes=((16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)),
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

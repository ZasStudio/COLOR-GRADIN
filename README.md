# COLOR-GRADIN

Copia el color grading de un video (o imagen) de referencia a otro video **usando máscaras**.

## Cómo funciona

Se hace en dos capas de máscaras suaves, igual que con qualifiers en DaVinci Resolve:

| Capa | Máscaras | Qué copia |
|---|---|---|
| 1. Luminancia | sombras · medios · altas luces | Contraste y el color de cada zona (p. ej. sombras teal / luces cálidas) |
| 2. Color | piel · cielo/azules · vegetación/verdes | El tono de piel, el cielo y los verdes de la referencia |

Cada zona del video se iguala con la misma zona de la referencia. Si una máscara no
aparece en los dos clips (por ejemplo, no hay cielo en uno de ellos), se omite.

Las máscaras sólo dependen del color de cada pixel, así que el grade completo se
guarda **exacto** en un LUT 3D `.cube` que puedes cargar en Resolve, Premiere,
Final Cut o CapCut y retocar allí.

## Uso

```bash
pip install -r requirements.txt
python color_match.py referencia.mp4 mi_video.mp4 -o salida/
```

La referencia también puede ser una imagen (`.jpg`, `.png`...).

Se genera en `salida/`:

- `mi_video_graded.mp4`: tu video ya corregido, con el audio original.
- `mi_video_grade.cube`: el LUT para usarlo en tu editor.
- `comparacion.jpg`: original, resultado y referencia lado a lado.
- `mascaras.jpg`: qué parte de la imagen toma cada máscara.

### Opciones

| Opción | Descripción |
|---|---|
| `-s 0.7` | Intensidad del grade (0 a 1) |
| `--sin-mascaras-color` | Usa sólo las máscaras de sombras, medios y altas luces |
| `--capas` | Exporta además un LUT por capa (color y luz de sombras, medios y altas luces) para montarlo con capas de ajuste en Premiere o Resolve |
| `--solo-lut` | Genera sólo el LUT y las previews, sin renderizar el video |
| `--frames 40` | Fotogramas que se analizan de cada clip (por defecto 24) |
| `--lut-size 33` | Resolución del LUT: 17, 33 o 65 (por defecto 65) |
| `--crf 18` | Calidad del video de salida (menor = mejor; por defecto 16) |

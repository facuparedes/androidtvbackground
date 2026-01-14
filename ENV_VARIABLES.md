# Environment Variables Documentation

## API Configuration

| Variable            | Default                  | Description                     |
| ------------------- | ------------------------ | ------------------------------- |
| `TMDB_BEARER_TOKEN` | Required                 | Your TMDB API Bearer token      |
| `TMDB_BASE_URL`     | `https://api.tmdb.org/3` | TMDB API base URL               |
| `TMDB_LANGUAGE`     | `en-US`                  | Language for metadata and logos |

## Content Settings

| Variable                 | Default           | Description                                      |
| ------------------------ | ----------------- | ------------------------------------------------ |
| `TMDB_NUMBER_OF_MOVIES`  | `5`               | Number of trending movies to process             |
| `TMDB_NUMBER_OF_TVSHOWS` | `5`               | Number of trending TV shows to process           |
| `MAX_CONTENT_AGE_DAYS`   | `90`              | Maximum age of content in days                   |
| `TMDB_CUSTOM_TEXT`       | `Now Trending on` | Custom text to display with TMDB logo            |
| `SHOW_CUSTOM_TEXT`       | `true`            | Set to `false` to hide custom text and TMDB logo |

## Output Settings

| Variable           | Default            | Description                                         |
| ------------------ | ------------------ | --------------------------------------------------- |
| `OUTPUT_DIR`       | `tmdb_backgrounds` | Directory for generated files                       |
| `PAGES_GITHUB_URL` | Empty              | Base URL for JSON file references                   |
| `CLEAN_OUTPUT_DIR` | `false`            | Set to `true` to delete all files before processing |
| `GENERATE_VIDEO`   | `true`             | Set to `false` to skip video generation             |

## Video Quality

| Variable                 | Default    | Description                                                 |
| ------------------------ | ---------- | ----------------------------------------------------------- |
| `TARGET_WIDTH`           | `1920`     | Output video width in pixels                                |
| `TARGET_HEIGHT`          | `1080`     | Output video height in pixels                               |
| `VIDEO_FPS`              | `24`       | Video framerate                                             |
| `VIDEO_DURATION`         | `15`       | Video duration in seconds                                   |
| `VIDEO_CRF`              | `23`       | Video quality (0-51, lower = better, 23 = good)             |
| `VIDEO_PRESET`           | `veryfast` | FFmpeg preset: `ultrafast`, `veryfast`, `fast`, `medium`    |
| `VIDEO_ZOOM`             | `1.08`     | Background zoom factor for slide effect (1.05-1.15)         |
| `VIDEO_PAN_START_MARGIN` | `0.1`      | Pan start position as fraction of available space (0.0-1.0) |
| `VIDEO_PAN_END_MARGIN`   | `0.9`      | Pan end position as fraction of available space (0.0-1.0)   |

**Preset Recommendations for CI:**

- `ultrafast` - Fastest encoding, larger files (~2-3x faster)
- `veryfast` - Balanced speed/quality (recommended)
- `fast` - Better quality, slower

## Image Quality

| Variable             | Default | Description                        |
| -------------------- | ------- | ---------------------------------- |
| `IMAGE_QUALITY`      | `85`    | JPEG quality (0-100)               |
| `BLUR_RADIUS`        | `40`    | Background blur intensity          |
| `VIGNETTE_STRENGTH`  | `2.5`   | Vignette effect strength (0-5)     |
| `MAX_OVERVIEW_LINES` | `8`     | Maximum lines for overview/summary |

## Performance Tuning

| Variable               | Default            | Description                                      |
| ---------------------- | ------------------ | ------------------------------------------------ |
| `MAX_WORKERS_DOWNLOAD` | `8`                | Parallel downloads (increase for faster network) |
| `MAX_WORKERS_PROCESS`  | `4`                | Parallel video processing                        |
| `FONT_PATH`            | `Roboto-Light.ttf` | Path to font file                                |

**Performance Tips:**

- For GitHub Actions (2-core): Use defaults
- For 4-core systems: `MAX_WORKERS_PROCESS=6`
- For 8+ core systems: `MAX_WORKERS_PROCESS=8`

## Content Filtering

| Variable             | Default | Description                                                 |
| -------------------- | ------- | ----------------------------------------------------------- |
| `EXCLUDED_COUNTRIES` | Empty   | Comma-separated ISO country codes (e.g., `jp,kr`)           |
| `EXCLUDED_GENRES`    | Empty   | Comma-separated genre names (e.g., `animation,documentary`) |
| `EXCLUDED_KEYWORDS`  | Empty   | Comma-separated keywords to exclude (e.g., `adult`)         |

## Example Configuration

```bash
# High Quality (slower)
VIDEO_CRF=20
VIDEO_PRESET=medium
IMAGE_QUALITY=95
MAX_WORKERS_PROCESS=2

# Fast Processing (lower quality)
VIDEO_CRF=26
VIDEO_PRESET=ultrafast
IMAGE_QUALITY=75
MAX_WORKERS_PROCESS=6

# Clean Start
CLEAN_OUTPUT_DIR=true
```

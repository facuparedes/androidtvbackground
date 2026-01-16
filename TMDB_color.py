import os
import shutil
import json
import textwrap
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from datetime import datetime, timedelta

import requests
import numpy as np
from rembg import remove, new_session
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageStat
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

TMDB_BEARER_TOKEN = os.getenv('TMDB_BEARER_TOKEN')
TMDB_BASE_URL = os.getenv('TMDB_BASE_URL') or 'https://api.tmdb.org/3'
PAGES_GITHUB_URL = (os.getenv("PAGES_GITHUB_URL") or "").rstrip('/')
LANGUAGE = os.getenv("TMDB_LANGUAGE") or "en-US"
REGION = os.getenv("TMDB_REGION") or ""
NUMBER_OF_MOVIES = int(os.getenv("TMDB_NUMBER_OF_MOVIES") or "5")
NUMBER_OF_TVSHOWS = int(os.getenv("TMDB_NUMBER_OF_TVSHOWS") or "5")
CUSTOM_TEXT = os.getenv("TMDB_CUSTOM_TEXT") or "Now Trending on"
SHOW_CUSTOM_TEXT = (os.getenv("SHOW_CUSTOM_TEXT") or "true").lower() == "true"

TARGET_WIDTH = int(os.getenv("TARGET_WIDTH") or "1920")
TARGET_HEIGHT = int(os.getenv("TARGET_HEIGHT") or "1080")
VIDEO_FPS = int(os.getenv("VIDEO_FPS") or "24")
VIDEO_DURATION = int(os.getenv("VIDEO_DURATION") or "15")
VIDEO_CRF = int(os.getenv("VIDEO_CRF") or "23")
VIDEO_PRESET = os.getenv("VIDEO_PRESET") or "veryfast"
VIDEO_ZOOM = float(os.getenv("VIDEO_ZOOM") or "1.08")
VIDEO_PAN_START_MARGIN = float(os.getenv("VIDEO_PAN_START_MARGIN") or "0.1")
VIDEO_PAN_END_MARGIN = float(os.getenv("VIDEO_PAN_END_MARGIN") or "0.9")

MAX_WORKERS_DOWNLOAD = int(os.getenv("MAX_WORKERS_DOWNLOAD") or "8")
MAX_WORKERS_PROCESS = int(os.getenv("MAX_WORKERS_PROCESS") or "4")
MAX_CONTENT_AGE_DAYS = int(os.getenv("MAX_CONTENT_AGE_DAYS") or "90")
CLEAN_OUTPUT_DIR = (os.getenv("CLEAN_OUTPUT_DIR") or "false").lower() == "true"
GENERATE_VIDEO = (os.getenv("GENERATE_VIDEO") or "true").lower() == "true"

BACKGROUND_DIR = os.getenv("OUTPUT_DIR") or "tmdb_backgrounds"
FONT_PATH = os.getenv("FONT_PATH") or "Roboto-Light.ttf"
FONT_PATH_REGULAR = os.getenv("FONT_PATH_REGULAR") or "Roboto-Regular.ttf"
FONT_URL = 'https://github.com/googlefonts/roboto/raw/main/src/hinted/Roboto-Light.ttf'

BLUR_RADIUS = int(os.getenv("BLUR_RADIUS") or "40")
IMAGE_QUALITY = int(os.getenv("IMAGE_QUALITY") or "85")
VIGNETTE_STRENGTH = float(os.getenv("VIGNETTE_STRENGTH") or "2.5")
MAX_OVERVIEW_LINES = int(os.getenv("MAX_OVERVIEW_LINES") or "4")
REMBG_SESSION = None

def get_session():
    global REMBG_SESSION
    if REMBG_SESSION is None:
        # u2netp is the lightweight model, much faster for this purpose
        try:
            REMBG_SESSION = new_session("u2netp")
        except Exception as e:
            print(f"Failed to load rembg session: {e}")
    return REMBG_SESSION


def parse_excluded_list(env_var_name):
    value = os.getenv(env_var_name, "")
    if not value:
        return []
    return [item.strip().lower() for item in value.split(",") if item.strip()]


EXCLUDED_COUNTRIES = parse_excluded_list("EXCLUDED_COUNTRIES")
EXCLUDED_KEYWORDS = parse_excluded_list("EXCLUDED_KEYWORDS")
EXCLUDED_GENRES = parse_excluded_list("EXCLUDED_GENRES")

MAX_AIR_DATE = datetime.now() - timedelta(days=MAX_CONTENT_AGE_DAYS)

session = requests.Session()
session.headers.update({
    "accept": "application/json",
    "Authorization": f"Bearer {TMDB_BEARER_TOKEN}"
})


def setup_environment():
    if CLEAN_OUTPUT_DIR and os.path.exists(BACKGROUND_DIR):
        shutil.rmtree(BACKGROUND_DIR)
        print(f"Cleaned output directory: {BACKGROUND_DIR}")
    
    os.makedirs(BACKGROUND_DIR, exist_ok=True)
    
    if not os.path.exists(FONT_PATH):
        try:
            r = session.get(FONT_URL, timeout=10)
            if r.status_code == 200:
                with open(FONT_PATH, 'wb') as f:
                    f.write(r.content)
                print("Font downloaded")
        except Exception as e:
            print(f"Font download failed: {e}")


def get_tmdb(endpoint, params=None):
    params = params or {}
    params['language'] = LANGUAGE
    if REGION:
        params['region'] = REGION
    try:
        response = session.get(f"{TMDB_BASE_URL}/{endpoint}", params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {}


def get_genres(media_type):
    data = get_tmdb(f'genre/{media_type}/list')
    return {g['id']: g['name'] for g in data.get('genres', [])}


def get_keywords(media_type, media_id):
    endpoint = f"{media_type}/{media_id}/keywords"
    data = get_tmdb(endpoint)
    results_key = 'results' if media_type == 'tv' else 'keywords'
    return [keyword['name'].lower() for keyword in data.get(results_key, [])]


def get_logo(media_type, media_id):
    lang_code = LANGUAGE.split("-")[0]
    data = get_tmdb(f"{media_type}/{media_id}/images", 
                    {'include_image_language': f'{lang_code},en,null'})
    
    logos = data.get('logos', [])
    if not logos:
        return None
    
    logos = [logo for logo in logos if not logo.get("file_path", "").lower().endswith('.svg')]
    if not logos:
        return None
    
    def get_best_logo(logo_list):
        return max(logo_list, key=lambda x: x.get("vote_average", 0))["file_path"]
    
    lang_logos = [logo for logo in logos if logo.get("iso_639_1") == lang_code]
    if lang_logos:
        return get_best_logo(lang_logos)
    
    en_logos = [logo for logo in logos if logo.get("iso_639_1") == "en"]
    if en_logos:
        return get_best_logo(en_logos)
    
    return get_best_logo(logos)


def is_content_too_old(item, media_type):
    date_key = 'release_date' if media_type == 'movie' else 'last_air_date'
    date_val = item.get(date_key)
    
    if media_type == 'tv' and not date_val:
        details = get_tmdb(f"tv/{item['id']}")
        date_val = details.get('last_air_date')

    if not date_val:
        return False
    
    try:
        content_date = datetime.strptime(date_val, "%Y-%m-%d")
        return content_date < MAX_AIR_DATE
    except ValueError:
        return False


def is_country_excluded(item):
    if not EXCLUDED_COUNTRIES:
        return False
    countries = [c.lower() for c in item.get('origin_country', [])]
    return any(country in EXCLUDED_COUNTRIES for country in countries)


def has_excluded_genre(item, genres_map):
    if not EXCLUDED_GENRES:
        return False
    item_genres = [genres_map.get(gid, '').lower() for gid in item.get('genre_ids', [])]
    return any(genre in EXCLUDED_GENRES for genre in item_genres)


def has_excluded_keyword(item, media_type):
    if not EXCLUDED_KEYWORDS:
        return False
    keywords = get_keywords(media_type, item['id'])
    return any(keyword in keywords for keyword in EXCLUDED_KEYWORDS)


def should_exclude(item, media_type, genres_map):
    return (is_content_too_old(item, media_type) or
            is_country_excluded(item) or
            has_excluded_genre(item, genres_map) or
            has_excluded_keyword(item, media_type))


def create_vignette(width, height):
    y, x = np.ogrid[0:height, 0:width]
    fade_ratio = 0.3
    fade_width = width * fade_ratio
    fade_height = height * fade_ratio
    
    dist_x = np.clip(x / fade_width, 0, 1)
    dist_y = np.clip((height - y - 150) / fade_height, 0, 1)
    alpha = (np.minimum(dist_x, dist_y) ** VIGNETTE_STRENGTH * 255).astype(np.uint8)
    
    return Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(radius=60))


# def generate_background_card(image):
#     # Redimensionar la imagen para cubrir todo el ancho sin difuminado
#     background = image.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
    
#     # Aplicar un ligero oscurecimiento para mejorar la legibilidad del texto
#     bg_array = np.array(background, dtype=np.float32)
#     bg_array = np.clip(bg_array * 0.6, 0, 255).astype(np.uint8)
    
#     canvas = Image.new("RGBA", (TARGET_WIDTH, TARGET_HEIGHT), (0, 0, 0, 255))
#     canvas.paste(Image.fromarray(bg_array), (0, 0))
    
#     return canvas.convert("RGB")

def get_subject_center_ai(image):
    try:
        # Resize for speed - u2net analysis doesn't need 4k
        analysis_width = 320
        scale = analysis_width / image.width
        analysis_height = int(image.height * scale)
        small_image = image.resize((analysis_width, analysis_height), Image.LANCZOS)
        
        session = get_session()
        if not session:
            return image.width / 2
            
        # Get the mask of the salient object
        # only_mask=True returns just the alpha channel
        mask = remove(small_image, session=session, only_mask=True)
        
        # Convert to numpy to find center of mass
        mask_arr = np.array(mask)
        
        # Find indices where mask > 0 (subject exists)
        # mask is grayscale (H, W) or (H, W, 3) depending on version, 
        # usually (H, W) for only_mask=True. Safely handle both.
        if len(mask_arr.shape) == 3:
            mask_arr = mask_arr[:, :, 0]
            
        y_indices, x_indices = np.nonzero(mask_arr > 10) # Threshold to avoid noise
        
        if len(x_indices) == 0:
            return image.width / 2 # No subject found
            
        # Calculate mean X position (center of mass horizontally)
        center_x_small = np.mean(x_indices)
        
        return center_x_small / scale
        
    except Exception as e:
        print(f"AI Subject detection error: {e}")
        return image.width / 2


def generate_background_card(image):
    blur_size = (max(TARGET_WIDTH // 20, 1), max(TARGET_HEIGHT // 20, 1))
    
    blurred = image.resize(blur_size, Image.LANCZOS)
    blurred = blurred.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
    background = blurred.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.BICUBIC)
    
    bg_array = np.array(background, dtype=np.float32)
    noise = np.random.uniform(-16, 16, bg_array.shape)
    bg_array = np.clip(bg_array * 0.4 + noise, 0, 255).astype(np.uint8)
    
    canvas = Image.new("RGBA", (TARGET_WIDTH, TARGET_HEIGHT), (0, 0, 0, 255))
    canvas.paste(Image.fromarray(bg_array), (0, 0))
    
    # Smart positioning logic with AI
    try:
        focal_x = get_subject_center_ai(image)
        scale_ratio = TARGET_HEIGHT / image.height
        overlay_height = TARGET_HEIGHT
        overlay_width = int(image.width * scale_ratio)
        
        # Ensure we have enough resolution
        overlay = image.resize((overlay_width, overlay_height), Image.LANCZOS).convert("RGBA")
        
        focal_x_scaled = focal_x * scale_ratio
        
        # Target: Center of right half (75% of screen width)
        target_center_x = TARGET_WIDTH * 0.75
        
        ideal_x = target_center_x - focal_x_scaled
        
        # Constraints
        # 1. Don't leave gap on RIGHT (Max X usually 0 or negative relative to right edge?)
        # width > target_width usually. 
        # If overlay_width < TARGET_WIDTH, we have a problem regardless (black bars).
        # Assuming overlay_width >= TARGET_WIDTH usually for backdrops? 
        # Actually backdrops are often landscape but might be slightly wider or narrower depending on aspect.
        # But we scaled by Height. 16:9 image scaled to 1080h will be 1920w.
        # If image is wider than 16:9, overlay_width > 1920.
        
        min_x = TARGET_WIDTH - overlay_width # Right edge alignment
        max_x = 0 # Left edge alignment (don't go beyond left edge? optional)
        
        # We want to pull left (negative x) as much as needed to center the focal point
        # But not more than min_x (which is negative or zero)
        
        overlay_x = int(max(ideal_x, min_x))
        
        # Also ensure we don't start too far right if image is super wide?
        # If ideal_x > 0, it means we have a gap on the left.
        # Background is blurred globally, so gap is fine visually (it shows blurred bg).
        # BUT user wanted "pegada al borde derecho" as fallback.
        
        vignette_mask = create_vignette(overlay_width, overlay_height)
        overlay.putalpha(vignette_mask)
        
        canvas.paste(overlay, (overlay_x, 0), overlay)
        
    except Exception as e:
        print(f"Error in AI positioning: {e}")
        # Fallback to simple right alignment
        scale_ratio = TARGET_HEIGHT / image.height
        overlay_height = TARGET_HEIGHT
        overlay_width = int(image.width * scale_ratio)
        overlay = image.resize((overlay_width, overlay_height), Image.LANCZOS).convert("RGBA")
        vignette_mask = create_vignette(overlay_width, overlay_height)
        overlay.putalpha(vignette_mask)
        overlay_x = TARGET_WIDTH - overlay_width
        canvas.paste(overlay, (overlay_x, 0), overlay)

    return canvas.convert("RGB")


def cleanup_temp_files(*file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)


def calculate_pan_coordinates(width, zoomed_width):
    pan_range = zoomed_width - width
    start_x = pan_range * VIDEO_PAN_START_MARGIN
    end_x = pan_range * VIDEO_PAN_END_MARGIN
    
    is_forward = np.random.choice([True, False])
    return (start_x, end_x) if is_forward else (end_x, start_x)


def build_ffmpeg_filter(width, height, x_start, x_end):
    total_frames = VIDEO_DURATION * VIDEO_FPS
    x_expr = f"if(eq(n,0),{x_start},{x_start}+({x_end}-{x_start})*n/{total_frames})"
    
    fade_duration = 2
    fade_out_start = VIDEO_DURATION - fade_duration - 0.1
    
    return (
        f"[0:v]crop=w={width}:h={height}:x='{x_expr}':y='(ih-oh)/2'[bg];"
        f"[bg][1:v]overlay=0:0:format=auto[combined];"
        f"[combined]fade=t=in:st=0:d={fade_duration}:color=0x000000,"
        f"fade=t=out:st={fade_out_start}:d={fade_duration}:color=0x000000[out]"
    )


def build_ffmpeg_command(temp_bg, temp_ov, filter_str, output_path):
    return [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(VIDEO_FPS),
        "-loop", "1", "-i", temp_bg,
        "-framerate", str(VIDEO_FPS),
        "-loop", "1", "-i", temp_ov,
        "-filter_complex", filter_str,
        "-map", "[out]",
        "-t", str(VIDEO_DURATION),
        "-c:v", "libx264",
        "-preset", VIDEO_PRESET,
        "-crf", str(VIDEO_CRF),
        "-maxrate", "2M",
        "-bufsize", "4M",
        "-pix_fmt", "yuv420p",
        "-vsync", "cfr",
        "-movflags", "+faststart",
        "-threads", "0",
        output_path
    ]


def create_video_ffmpeg(bg_image, overlay_image, output_path):
    temp_bg = output_path.replace(".mp4", "_temp_bg.png")
    temp_ov = output_path.replace(".mp4", "_temp_ov.png")
    
    try:
        width, height = bg_image.size
        zoomed_size = (int(width * VIDEO_ZOOM), int(height * VIDEO_ZOOM))
        
        bg_image.resize(zoomed_size, Image.LANCZOS).save(temp_bg, optimize=False)
        overlay_image.save(temp_ov, optimize=False)
        
        x_start, x_end = calculate_pan_coordinates(width, zoomed_size[0])
        filter_str = build_ffmpeg_filter(width, height, x_start, x_end)
        cmd = build_ffmpeg_command(temp_bg, temp_ov, filter_str, output_path)
        
        subprocess.run(cmd, capture_output=True, check=True, timeout=120)
        return True
    
    except subprocess.TimeoutExpired:
        print(f"FFmpeg timeout: {output_path}")
        return False
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        return False
    except Exception as e:
        print(f"Video creation error: {e}")
        return False
    finally:
        cleanup_temp_files(temp_bg, temp_ov)


def clean_filename(s):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)


def download_image(url, timeout=10):
    try:
        response = session.get(url, timeout=timeout)
        if response.status_code != 200:
            return None
        
        image_data = BytesIO(response.content)
        image = Image.open(image_data)
        image.verify()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Failed to download/validate image from {url}: {e}")
        return None


def draw_text_with_shadow(draw, position, text, font, shadow_offset=2):
    draw.text((position[0] + shadow_offset, position[1] + shadow_offset), 
             text, font=font, fill="black")
    draw.text(position, text, font=font, fill="white")


def scale_value(value, scale_factor):
    return int(value * scale_factor)


def create_scaled_font(font_path, size, scale_factor):
    return ImageFont.truetype(font_path, size=scale_value(size, scale_factor))


def fetch_media_details(item, media_type):
    item_id = item['id']
    details = get_tmdb(f"{media_type}/{item_id}")
    
    if media_type == 'movie':
        runtime = details.get('runtime', 0)
        return f"{runtime//60}h{runtime%60}min" if runtime else "N/A"
    
    season_count = details.get('number_of_seasons', 0)
    season_label = "Season" if season_count == 1 else "Seasons"
    return f"{season_count} {season_label}" if season_count else "N/A"


def build_metadata_text(item, media_type, genres_map):
    year = (item.get('release_date') or item.get('first_air_date') or "N/A")[:4]
    rating = round(item.get('vote_average', 0), 1)
    genre_names = [genres_map[gid] for gid in item.get('genre_ids', []) if gid in genres_map]
    genre_str = ', '.join(genre_names)
    extra_info = fetch_media_details(item, media_type)
    return f"{genre_str}  •  {year}  •  {extra_info}  •  TMDB: {rating}"


def wrap_overview_text(overview, max_lines=MAX_OVERVIEW_LINES, width=65):
    wrapped = "\n".join(
        textwrap.wrap(overview, width=width, max_lines=max_lines, placeholder=" ...")
    )
    return wrapped


def ensure_logo_contrast(image):
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    grayscale = image.convert("L")
    a = image.split()[3]
    
    stat = ImageStat.Stat(grayscale, mask=a)
    if stat.count[0] == 0:
        return image
        
    avg_brightness = stat.mean[0]
    
    # If brightness is low (dark logo), make it white
    if avg_brightness < 60:
        white = Image.new('RGBA', image.size, (255, 255, 255, 0))
        white.putalpha(a)
        return white
        
    return image


def add_logo_to_overlay(overlay, media_type, item_id, padding_x, info_position, scale_factor, padding_y_logo_to_info):
    logo_path = get_logo(media_type, item_id)
    if not logo_path:
        return False
    
    try:
        logo_image = download_image(f"https://image.tmdb.org/t/p/original{logo_path}")
        if not logo_image:
            return False

        # Ensure contrast (make white if too dark)
        logo_image = ensure_logo_contrast(logo_image)
        
        # Dimensiones objetivo para el logo
        target_width = scale_value(800, scale_factor)
        max_width = scale_value(1200, scale_factor)
        max_height = scale_value(600, scale_factor)
        
        # Calcular el tamaño escalado manteniendo aspect ratio
        aspect_ratio = logo_image.width / logo_image.height
        
        # Si el logo es más ancho que alto (horizontal)
        if aspect_ratio > 1.5:
            new_width = min(max_width, max(target_width, logo_image.width))
            new_height = int(new_width / aspect_ratio)
        else:
            # Para logos más cuadrados o verticales
            new_height = min(max_height, int(target_width / aspect_ratio))
            new_width = int(new_height * aspect_ratio)
        
        # Asegurar que no exceda los límites máximos
        if new_width > max_width:
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
        if new_height > max_height:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
        
        logo_image = logo_image.resize((new_width, new_height), Image.LANCZOS)
        logo_y = info_position[1] - logo_image.height - padding_y_logo_to_info
        
        # Add shadow to ensure legibility
        shadow = Image.new('RGBA', logo_image.size, (0, 0, 0, 160))
        shadow.putalpha(logo_image.split()[3])
        # Paste shadow with small offset (scaled if possible, or fixed)
        overlay.paste(shadow, (padding_x + 2, logo_y + 2), shadow)

        logo_rgba = logo_image.convert('RGBA')
        overlay.paste(logo_rgba, (padding_x, logo_y), logo_rgba)
        return True
    except Exception as e:
        return False


def add_tmdb_branding(overlay, custom_position, scale_factor):
    tmdb_logo_path = os.path.join(os.path.dirname(__file__), "tmdblogo.png")
    if not os.path.exists(tmdb_logo_path):
        return
    
    try:
        tmdb_logo = Image.open(tmdb_logo_path).convert('RGBA')
        max_tmdb_logo_size = (scale_value(90, scale_factor), scale_value(90, scale_factor))
        tmdb_logo.thumbnail(max_tmdb_logo_size, Image.LANCZOS)
        tmdb_logo_position = (scale_value(510, scale_factor), 
                            custom_position[1] + scale_value(28, scale_factor))
        
        # Crear sombra del logo
        shadow_offset = 2
        shadow = Image.new('RGBA', tmdb_logo.size, (0, 0, 0, 0))
        # Usar el alpha del logo para crear la sombra negra
        shadow_data = np.array(tmdb_logo)
        shadow_array = np.zeros_like(shadow_data)
        shadow_array[:, :, 3] = shadow_data[:, :, 3]  # Copiar el canal alpha
        shadow = Image.fromarray(shadow_array)
        
        # Pegar primero la sombra
        shadow_position = (tmdb_logo_position[0] + shadow_offset, tmdb_logo_position[1] + shadow_offset)
        overlay.paste(shadow, shadow_position, shadow)
        
        # Luego pegar el logo
        overlay.paste(tmdb_logo, tmdb_logo_position, tmdb_logo)
    except Exception:
        pass


def create_media_overlay(item, media_type, genres_map, background_size, scale_factor):
    overlay = Image.new("RGBA", background_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    font_title = create_scaled_font(FONT_PATH, 190, scale_factor)
    font_info = create_scaled_font(FONT_PATH_REGULAR, 60, scale_factor)
    font_overview = create_scaled_font(FONT_PATH, 60, scale_factor)
    
    padding_x = scale_value(210, scale_factor)
    padding_y_logo_to_info = scale_value(100, scale_factor)
    padding_y_info_to_summary = scale_value(50, scale_factor)
    padding_y_summary_to_custom = scale_value(50, scale_factor)
    
    title_position = (scale_value(200, scale_factor), scale_value(520, scale_factor))
    
    name = item.get('title') if media_type == 'movie' else item.get('name')
    meta_text = build_metadata_text(item, media_type, genres_map)
    wrapped_overview = wrap_overview_text(item.get('overview', ''))
    
    info_bbox = draw.textbbox((0, 0), meta_text, font=font_info)
    info_height = info_bbox[3] - info_bbox[1]
    
    overview_bbox = draw.textbbox((0, 0), wrapped_overview, font=font_overview)
    summary_height = overview_bbox[3] - overview_bbox[1]
    
    base_y = scale_value(750, scale_factor)
    info_position = (padding_x, base_y)
    overview_position = (padding_x, base_y + info_height + padding_y_info_to_summary)
    custom_position = (padding_x, overview_position[1] + summary_height + padding_y_summary_to_custom)
    
    logo_drawn = add_logo_to_overlay(overlay, media_type, item['id'], padding_x, info_position, scale_factor, padding_y_logo_to_info)
    
    if not logo_drawn:
        draw_text_with_shadow(draw, title_position, name, font_title)
    
    draw_text_with_shadow(draw, info_position, meta_text, font_info)
    draw_text_with_shadow(draw, overview_position, wrapped_overview, font_overview)
    
    if SHOW_CUSTOM_TEXT:
        draw_text_with_shadow(draw, custom_position, CUSTOM_TEXT, font_overview)
        add_tmdb_branding(overlay, custom_position, scale_factor)
    
    return overlay


def create_json_metadata(name, clean_title, image_file, video_file, base_url):
    return {
        "location": "TMDB",
        "title": name,
        "author": "TMDB",
        "url_img": f"{base_url}/{clean_title}.jpg" if os.path.exists(image_file) else None,
        "url_1080p": f"{base_url}/{clean_title}.mp4" if os.path.exists(video_file) else None,
        "url_4k": f"{base_url}/{clean_title}.mp4" if os.path.exists(video_file) else None,
        "url_1080p_hdr": None,
        "url_4k_hdr": None
    }


def save_final_image(background, overlay, output_path):
    final_image = Image.alpha_composite(background.convert("RGBA"), overlay).convert("RGB")
    final_image.save(output_path, quality=IMAGE_QUALITY, optimize=True, progressive=True)


def process_media(item, media_type, genres_map):
    name = item.get('title') if media_type == 'movie' else item.get('name')
    if not name:
        return None
    
    if should_exclude(item, media_type, genres_map):
        print(f"Excluded: {name}")
        return None

    backdrop = item.get('backdrop_path')
    if not backdrop:
        return None
    
    clean_title = clean_filename(name)
    image_file = os.path.join(BACKGROUND_DIR, f"{clean_title}.jpg")
    video_file = os.path.join(BACKGROUND_DIR, f"{clean_title}.mp4")
    base_url = PAGES_GITHUB_URL or "."
    
    if os.path.exists(image_file) and (not GENERATE_VIDEO or os.path.exists(video_file)):
        json_data = create_json_metadata(name, clean_title, image_file, video_file, base_url)
        return clean_title, json_data

    try:
        source_image = download_image(f"https://image.tmdb.org/t/p/original{backdrop}")
        if not source_image:
            return None
        
        background = generate_background_card(source_image)
        scale_factor = TARGET_HEIGHT / 2160.0
        overlay = create_media_overlay(item, media_type, genres_map, background.size, scale_factor)
        
        save_final_image(background, overlay, image_file)
        
        if GENERATE_VIDEO and not os.path.exists(video_file):
            video_success = create_video_ffmpeg(background, overlay, video_file)
            if not video_success:
                print(f"Video failed: {name}")

        print(f"Processed: {name}")
        json_data = create_json_metadata(name, clean_title, image_file, video_file, base_url)
        return clean_title, json_data

    except Exception as e:
        print(f"Error {name}: {e}")
        return None


def filter_valid_items(items, media_type, genres_map, max_count):
    valid_items = []
    name_key = 'title' if media_type == 'movie' else 'name'
    
    for item in items:
        if not should_exclude(item, media_type, genres_map):
            valid_items.append(item)
            if len(valid_items) >= max_count:
                break
        else:
            print(f"Excluded: {item.get(name_key, 'Unknown')}")
    
    return valid_items


def submit_processing_tasks(executor, valid_movies, valid_tv, movie_genres, tv_genres):
    tasks = []
    for movie in valid_movies:
        tasks.append(executor.submit(process_media, movie, 'movie', movie_genres))
    for tv_show in valid_tv:
        tasks.append(executor.submit(process_media, tv_show, 'tv', tv_genres))
    return tasks


def collect_results(tasks):
    valid_files = set()
    json_output = []
    
    for future in tqdm(as_completed(tasks), total=len(tasks), desc="Processing"):
        try:
            result = future.result()
            if result:
                clean_name, json_data = result
                valid_files.add(f"{clean_name}.jpg")
                valid_files.add(f"{clean_name}.mp4")
                json_output.append(json_data)
        except Exception as e:
            print(f"Task error: {e}")
    
    return valid_files, json_output


def save_json_output(json_output):
    json_file = os.path.join(BACKGROUND_DIR, "videos.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=4, ensure_ascii=False)
    print(f"Saved: {json_file}")


def cleanup_old_files(valid_files):
    for filename in os.listdir(BACKGROUND_DIR):
        if filename == 'videos.json':
            continue
        if (filename.endswith('.jpg') or filename.endswith('.mp4')) and filename not in valid_files:
            try:
                os.remove(os.path.join(BACKGROUND_DIR, filename))
                print(f"Cleaned: {filename}")
            except Exception:
                pass


def main():
    setup_environment()
    
    print("Fetching trending content...")
    
    fetch_multiplier = 3
    max_fetch_limit = 20
    max_movies = min(NUMBER_OF_MOVIES * fetch_multiplier, max_fetch_limit)
    max_tv = min(NUMBER_OF_TVSHOWS * fetch_multiplier, max_fetch_limit)
    
    all_movies = get_tmdb('trending/movie/week').get('results', [])[:max_movies]
    all_tv = get_tmdb('trending/tv/week').get('results', [])[:max_tv]
    
    if REGION:
        print(f"Filtering content available in region: {REGION}")
    
    movie_genres = get_genres('movie')
    tv_genres = get_genres('tv')
    
    valid_movies = filter_valid_items(all_movies, 'movie', movie_genres, NUMBER_OF_MOVIES)
    valid_tv = filter_valid_items(all_tv, 'tv', tv_genres, NUMBER_OF_TVSHOWS)
    
    print(f"Valid content: {len(valid_movies)} movies, {len(valid_tv)} TV shows")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_PROCESS) as executor:
        tasks = submit_processing_tasks(executor, valid_movies, valid_tv, movie_genres, tv_genres)
        valid_files, json_output = collect_results(tasks)

    save_json_output(json_output)
    cleanup_old_files(valid_files)
    
    print("Done")


if __name__ == "__main__":
    main()

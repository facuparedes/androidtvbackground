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
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

TMDB_BEARER_TOKEN = os.getenv('TMDB_BEARER_TOKEN')
TMDB_BASE_URL = os.getenv('TMDB_BASE_URL') or 'https://api.tmdb.org/3'
PAGES_GITHUB_URL = (os.getenv("PAGES_GITHUB_URL") or "").rstrip('/')
LANGUAGE = os.getenv("TMDB_LANGUAGE") or "en-US"
NUMBER_OF_MOVIES = int(os.getenv("TMDB_NUMBER_OF_MOVIES") or "5")
NUMBER_OF_TVSHOWS = int(os.getenv("TMDB_NUMBER_OF_TVSHOWS") or "5")

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

BACKGROUND_DIR = os.getenv("OUTPUT_DIR") or "tmdb_backgrounds"
FONT_PATH = os.getenv("FONT_PATH") or "Roboto-Light.ttf"
FONT_URL = 'https://github.com/googlefonts/roboto/raw/main/src/hinted/Roboto-Light.ttf'

BLUR_RADIUS = int(os.getenv("BLUR_RADIUS") or "40")
IMAGE_QUALITY = int(os.getenv("IMAGE_QUALITY") or "85")
VIGNETTE_STRENGTH = float(os.getenv("VIGNETTE_STRENGTH") or "2.5")

EXCLUDED_COUNTRIES = os.getenv("EXCLUDED_COUNTRIES", "").split(",") if os.getenv("EXCLUDED_COUNTRIES") else []
EXCLUDED_KEYWORDS = os.getenv("EXCLUDED_KEYWORDS", "").split(",") if os.getenv("EXCLUDED_KEYWORDS") else []
EXCLUDED_GENRES = {}

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
    try:
        r = session.get(f"{TMDB_BASE_URL}/{endpoint}", params=params, timeout=10)
        return r.json() if r.status_code == 200 else {}
    except:
        return {}


def get_genres(media_type):
    data = get_tmdb(f'genre/{media_type}/list')
    return {g['id']: g['name'] for g in data.get('genres', [])}


def get_keywords(media_type, media_id):
    endpoint = f"tv/{media_id}/keywords" if media_type == 'tv' else f"movie/{media_id}/keywords"
    key = 'results' if media_type == 'tv' else 'keywords'
    data = get_tmdb(endpoint)
    return [k['name'].lower() for k in data.get(key, [])]


def get_logo(media_type, media_id):
    data = get_tmdb(f"{media_type}/{media_id}/images", {'include_image_language': f'{LANGUAGE},en,null'})
    logos = data.get('logos', [])
    if not logos:
        return None
    
    match = [l for l in logos if l.get("iso_639_1") == LANGUAGE]
    if match:
        return sorted(match, key=lambda x: x.get("vote_average", 0), reverse=True)[0]["file_path"]
    
    return sorted(logos, key=lambda x: x.get("vote_average", 0), reverse=True)[0]["file_path"]


def should_exclude(item, media_type, genres_map):
    date_key = 'release_date' if media_type == 'movie' else 'last_air_date'
    date_val = item.get(date_key)
    
    if media_type == 'tv' and not date_val:
        details = get_tmdb(f"tv/{item['id']}")
        date_val = details.get('last_air_date')

    if date_val:
        try:
            dt = datetime.strptime(date_val, "%Y-%m-%d")
            if dt < MAX_AIR_DATE:
                return True
        except:
            pass
            
    countries = [c.lower() for c in item.get('origin_country', [])]
    item_genres = [genres_map.get(gid, '') for gid in item.get('genre_ids', [])]
    
    for c in countries:
        if c in EXCLUDED_COUNTRIES:
            bad_genres = EXCLUDED_GENRES.get(c, [])
            if '*' in bad_genres or any(g in bad_genres for g in item_genres):
                return True
                
    if EXCLUDED_KEYWORDS:
        keys = get_keywords(media_type, item['id'])
        if any(k in keys for k in EXCLUDED_KEYWORDS):
            return True
            
    return False


def create_vignette(w, h):
    y, x = np.ogrid[0:h, 0:w]
    fade_ratio = 0.3
    rx, ry = w * fade_ratio, h * fade_ratio
    
    dist_x = np.clip(x / rx, 0, 1)
    dist_y = np.clip((h - y - 150) / ry, 0, 1)
    alpha = (np.minimum(dist_x, dist_y) ** VIGNETTE_STRENGTH * 255).astype(np.uint8)
    
    mask = Image.fromarray(alpha)
    return mask.filter(ImageFilter.GaussianBlur(radius=60))


def generate_background_card(img_pil):
    bg_w, bg_h = TARGET_WIDTH, TARGET_HEIGHT
    small_w = max(bg_w // 20, 1)
    small_h = max(bg_h // 20, 1)
    
    bg_small = img_pil.resize((small_w, small_h), Image.LANCZOS)
    bg_small = bg_small.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
    bg_base = bg_small.resize((bg_w, bg_h), Image.BICUBIC)
    
    arr = np.array(bg_base, dtype=np.float32)
    noise = np.random.uniform(-16, 16, arr.shape)
    arr = np.clip(arr * 0.4 + noise, 0, 255).astype(np.uint8)
    
    canvas = Image.new("RGBA", (bg_w, bg_h), (0, 0, 0, 255))
    canvas.paste(Image.fromarray(arr), (0, 0))
    
    target_img_w = int(TARGET_WIDTH * 0.78)
    ratio = target_img_w / img_pil.width
    new_h = int(img_pil.height * ratio)
    img_resized = img_pil.resize((target_img_w, new_h), Image.LANCZOS).convert("RGBA")
    
    mask = create_vignette(target_img_w, img_resized.height)
    img_resized.putalpha(mask)
    
    canvas.paste(img_resized, (TARGET_WIDTH - target_img_w, 0), img_resized)
    return canvas.convert("RGB")


def create_video_ffmpeg(bg_image, overlay_image, output_path):
    temp_bg = output_path.replace(".mp4", "_temp_bg.png")
    temp_ov = output_path.replace(".mp4", "_temp_ov.png")
    
    try:
        w, h = bg_image.size
        zoom = VIDEO_ZOOM
        zw, zh = int(w * zoom), int(h * zoom)
        
        # PNG para evitar artifacts de compresión en el movimiento
        bg_image.resize((zw, zh), Image.LANCZOS).save(temp_bg, optimize=False)
        overlay_image.save(temp_ov, optimize=False)
        
        diff_w = zw - w
        # Márgenes configurables para controlar la distancia del movimiento
        min_x = diff_w * VIDEO_PAN_START_MARGIN
        max_x = diff_w * VIDEO_PAN_END_MARGIN
        dist = max_x - min_x
        
        # Dirección aleatoria del pan
        direction = np.random.choice([True, False])
        x_start = min_x if direction else max_x
        x_end = max_x if direction else min_x
        
        total_frames = VIDEO_DURATION * VIDEO_FPS
        
        # Crop animado basado en frames (n) para movimiento perfectamente lineal
        # Usa 'n' (frame number) en lugar de 't' (tiempo) para evitar interpolación
        x_expr = f"if(eq(n,0),{x_start},{x_start}+({x_end}-{x_start})*n/{total_frames})"
        
        filter_str = (
            f"[0:v]crop=w={w}:h={h}:x='{x_expr}':y='(ih-oh)/2'[bg];"
            f"[bg][1:v]overlay=0:0:format=auto[combined];"
            f"[combined]fade=t=in:st=0:d=2:color=0x000000,fade=t=out:st={VIDEO_DURATION-2.1}:d=2:color=0x000000[out]"
        )
        
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(VIDEO_FPS),  # FPS en el input para timing correcto
            "-loop", "1", "-i", temp_bg,
            "-framerate", str(VIDEO_FPS),
            "-loop", "1", "-i", temp_ov,
            "-filter_complex", filter_str,
            "-map", "[out]",
            "-t", str(VIDEO_DURATION),
            "-c:v", "libx264",
            "-preset", VIDEO_PRESET,
            "-crf", str(VIDEO_CRF),
            "-pix_fmt", "yuv420p",
            "-vsync", "cfr",  # Frame rate constante - CRÍTICO para evitar stuttering
            "-movflags", "+faststart",
            "-threads", "0",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, check=True, timeout=120)
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
        for f in [temp_bg, temp_ov]:
            if os.path.exists(f):
                os.remove(f)


def clean_filename(s):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)


def download_image(url, timeout=10):
    try:
        resp = session.get(url, timeout=timeout)
        if resp.status_code == 200:
            return Image.open(BytesIO(resp.content))
    except:
        pass
    return None


def fetch_media_details(item, media_type):
    item_id = item['id']
    if media_type == 'movie':
        d = get_tmdb(f"movie/{item_id}")
        r = d.get('runtime', 0)
        return f"{r//60}h{r%60}min" if r else "N/A"
    else:
        d = get_tmdb(f"tv/{item_id}")
        cnt = d.get('number_of_seasons', 0)
        return f"{cnt} Season{'s' if cnt != 1 else ''}" if cnt else "N/A"


def process_media(item, media_type, genres_map):
    item_id = item['id']
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
    f_img = os.path.join(BACKGROUND_DIR, f"{clean_title}.jpg")
    f_vid = os.path.join(BACKGROUND_DIR, f"{clean_title}.mp4")
    
    base = PAGES_GITHUB_URL if PAGES_GITHUB_URL else "."
    json_data = {
        "location": "TMDB",
        "title": name,
        "author": "TMDB",
        "url_img": f"{base}/{clean_title}.jpg",
        "url_1080p": f"{base}/{clean_title}.mp4",
        "url_4k": f"{base}/{clean_title}.mp4",
        "url_1080p_hdr": None,
        "url_4k_hdr": None
    }

    if os.path.exists(f_img) and os.path.exists(f_vid):
        return clean_title, json_data

    try:
        src_img = download_image(f"https://image.tmdb.org/t/p/original{backdrop}")
        if not src_img:
            return None
        
        bg_card = generate_background_card(src_img)
        overlay = Image.new("RGBA", bg_card.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        s_factor = TARGET_HEIGHT / 2160.0
        font_t = ImageFont.truetype(FONT_PATH, size=int(190 * s_factor))
        font_o = ImageFont.truetype(FONT_PATH, size=int(50 * s_factor))
        
        padding_x = int(210 * s_factor)
        
        year = (item.get('release_date') or item.get('first_air_date') or "N/A")[:4]
        rating = round(item.get('vote_average', 0), 1)
        genre_str = ', '.join([genres_map.get(g, '') for g in item.get('genre_ids', []) if g in genres_map])
        extra = fetch_media_details(item, media_type)
        meta_text = f"{genre_str}  •  {year}  •  {extra}  •  TMDB: {rating}"
        
        wrap_width = 65
        overview_lines = textwrap.wrap(item.get('overview', ''), width=wrap_width, max_lines=8, placeholder=" ...")
        
        logo_path = get_logo(media_type, item_id)
        logo_drawn = False
        current_y = int(420 * s_factor)
        
        if logo_path:
            l_img = download_image(f"https://image.tmdb.org/t/p/original{logo_path}")
            if l_img:
                lw, lh = int(1000 * s_factor), int(500 * s_factor)
                l_img.thumbnail((lw, lh), Image.LANCZOS)
                overlay.paste(l_img.convert('RGBA'), (padding_x, current_y), l_img.convert('RGBA'))
                current_y += l_img.height + int(25 * s_factor)
                logo_drawn = True
            
        if not logo_drawn:
            draw.text((padding_x + 2, current_y + 2), name, font=font_t, fill="black")
            draw.text((padding_x, current_y), name, font=font_t, fill="white")
            bbox = font_t.getbbox(name)
            current_y += (bbox[3] - bbox[1]) + int(25 * s_factor)

        current_y += int(50 * s_factor)
        draw.text((padding_x + 2, current_y + 2), meta_text, font=font_o, fill="black")
        draw.text((padding_x, current_y), meta_text, font=font_o, fill="white")
        current_y += int(80 * s_factor)

        for line in overview_lines:
            draw.text((padding_x + 2, current_y + 2), line, font=font_o, fill="black")
            draw.text((padding_x, current_y), line, font=font_o, fill="white")
            bbox = font_o.getbbox(line)
            current_y += (bbox[3] - bbox[1]) + int(10 * s_factor)

        final = Image.alpha_composite(bg_card.convert("RGBA"), overlay).convert("RGB")
        final.save(f_img, quality=IMAGE_QUALITY, optimize=True)
        
        if not os.path.exists(f_vid):
            is_ok = create_video_ffmpeg(bg_card, overlay, f_vid)
            if not is_ok:
                print(f"Video failed: {name}")

        print(f"Processed: {name}")
        return clean_title, json_data

    except Exception as e:
        print(f"Error {name}: {e}")
        return None


def main():
    setup_environment()
    
    print("Fetching trending content...")
    movies_raw = get_tmdb('trending/movie/week').get('results', [])[:NUMBER_OF_MOVIES]
    tv_raw = get_tmdb('trending/tv/week').get('results', [])[:NUMBER_OF_TVSHOWS]
    
    movie_genres = get_genres('movie')
    tv_genres = get_genres('tv')
    
    tasks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_PROCESS) as executor:
        for m in movies_raw:
            tasks.append(executor.submit(process_media, m, 'movie', movie_genres))
        for t in tv_raw:
            tasks.append(executor.submit(process_media, t, 'tv', tv_genres))
            
        valid_files = set()
        json_output = []
        
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Processing"):
            try:
                result = future.result()
                if result:
                    clean_name, j_data = result
                    valid_files.add(f"{clean_name}.jpg")
                    valid_files.add(f"{clean_name}.mp4")
                    json_output.append(j_data)
            except Exception as e:
                print(f"Task error: {e}")

    json_file = os.path.join(BACKGROUND_DIR, "videos.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=4, ensure_ascii=False)
    print(f"Saved: {json_file}")
        
    for filename in os.listdir(BACKGROUND_DIR):
        if (filename.endswith('.jpg') or filename.endswith('.mp4')) and filename not in valid_files:
            try:
                os.remove(os.path.join(BACKGROUND_DIR, filename))
                print(f"Cleaned: {filename}")
            except:
                pass
            
    print("Done")


if __name__ == "__main__":
    main()

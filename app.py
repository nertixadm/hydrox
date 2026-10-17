import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
from hydrox_analyzer import *
import tempfile
import os
import io
import zipfile
import math
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from matplotlib import font_manager
import hashlib
from collections import Counter
import re
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Анализ чата Telegram")
config = load_config("config.yaml")
colors = config['colors']
session_gap = config['analysis']['session_gap_hours']
top_n_default = config['analysis']['top_n_words']
min_len = config['analysis']['min_word_length']
stop_words = set(config['analysis']['stop_words'])
stop_words.add('это')
top_bigrams = config['analysis']['top_n_bigrams']

# ---------- СКЛОНЕНИЕ ЧИСЛИТЕЛЬНЫХ ----------
def _pluralize(n, one, few, many):
    n = abs(int(n))
    n_mod100 = n % 100
    n_mod10 = n % 10
    if 11 <= n_mod100 <= 14:
        return many
    if n_mod10 == 1:
        return one
    if 2 <= n_mod10 <= 4:
        return few
    return many

# ---------- ШРИФТ ----------
def _load_font(size, bold=False):
    candidates = [
        ('Noto Sans', 'bold' if bold else 'regular'),
        ('DejaVu Sans', 'bold' if bold else 'normal'),
        ('Arial', 'bold' if bold else 'regular'),
    ]
    for family, weight in candidates:
        try:
            fp = font_manager.FontProperties(family=family, weight=weight)
            path = font_manager.findfont(fp)
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()

# ---------- ОЧИСТКА ИМЕНИ ----------
def _clean_name(name):
    if not name:
        return "Unknown"
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937\U00010000-\U0010ffff\u2640-\u2642"
        "\u2600-\u2B55\u200d\u23cf\u23e9\u231a\ufe0f\u3030"
        "]+", flags=re.UNICODE
    )
    cleaned = emoji_pattern.sub('', name).strip()
    if not cleaned or cleaned in ['+', '×', 'z', '□']:
        return re.sub(r'[^\w\s\-\.@_+]', '_', name)[:15] or "User"
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned[:20] if cleaned else "User"

# ---------- УТИЛИТЫ РИСОВАНИЯ ----------
def _gradient_bg(size, c1, c2, angle=135):
    img = Image.new('RGB', size)
    draw = ImageDraw.Draw(img)
    w, h = size
    rad = math.radians(angle)
    dx, dy = math.cos(rad), math.sin(rad)
    denom = w * abs(dx) + h * abs(dy)
    for y in range(h):
        for x in range(w):
            t = (x * dx + y * dy) / denom if denom != 0 else 0
            t = max(0.0, min(1.0, t))
            r = int(c1[0] + (c2[0] - c1[0]) * t)
            g = int(c1[1] + (c2[1] - c1[1]) * t)
            b = int(c1[2] + (c2[2] - c1[2]) * t)
            draw.point((x, y), fill=(r, g, b))
    return img

def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _draw_glow_text(img, xy, text, font, fill, glow_color, glow_radius=8):
    draw = ImageDraw.Draw(img)
    try:
        bbox = draw.textbbox((0, 0), text, font=font, anchor='mm')
    except TypeError:
        bbox = draw.textbbox((0, 0), text, font=font)
    w = max(1, int(bbox[2] - bbox[0]))
    h = max(1, int(bbox[3] - bbox[1]))
    pad = glow_radius * 4 + 20
    tmp_w = int(w + pad * 2)
    tmp_h = int(h + pad * 2)
    tmp = Image.new('RGBA', (tmp_w, tmp_h), (0, 0, 0, 0))
    tmp_draw = ImageDraw.Draw(tmp)
    glow_rgba = (*glow_color, 255) if len(glow_color) == 3 else glow_color
    try:
        tmp_draw.text((tmp_w // 2, tmp_h // 2), text, font=font, fill=glow_rgba, anchor='mm')
    except TypeError:
        tmp_draw.text((tmp_w // 2, tmp_h // 2), text, font=font, fill=glow_rgba)
    if glow_radius > 0:
        tmp = tmp.filter(ImageFilter.GaussianBlur(glow_radius))
    paste_x = xy[0] - tmp_w // 2
    paste_y = xy[1] - tmp_h // 2
    img.paste(tmp, (paste_x, paste_y), tmp)
    try:
        draw.text(xy, text, font=font, fill=fill, anchor='mm')
    except TypeError:
        draw.text(xy, text, font=font, fill=fill)

def _draw_rounded_rect(draw, xy, radius, fill):
    draw.rounded_rectangle(xy, radius=radius, fill=fill)

def _draw_orb(img, cx, cy, r, color, alpha=80):
    tmp = Image.new('RGBA', img.size, (0, 0, 0, 0))
    tmp_draw = ImageDraw.Draw(tmp)
    tmp_draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(*color, alpha))
    tmp = tmp.filter(ImageFilter.GaussianBlur(r // 2))
    img.paste(tmp, (0, 0), tmp)

def _format_number(n):
    return f"{n:,}".replace(',', ' ')

def _format_timedelta(td):
    total_seconds = int(td.total_seconds())
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    parts = []
    if days:
        parts.append(f"{days} {_pluralize(days, 'день', 'дня', 'дней')}")
    if hours:
        parts.append(f"{hours} {_pluralize(hours, 'час', 'часа', 'часов')}")
    if minutes:
        parts.append(f"{minutes} {_pluralize(minutes, 'минута', 'минуты', 'минут')}")
    return ' '.join(parts) if parts else "менее минуты"

# ---------- АНАЛИТИЧЕСКИЕ ФУНКЦИИ ----------
def _find_longest_dialog(df, gap_minutes=30):
    df_sorted = df.sort('date')
    best_start = None
    best_end = None
    best_duration = timedelta(0)
    best_count = 0

    session_start = None
    session_end = None
    session_count = 0
    prev_time = None

    for row in df_sorted.iter_rows(named=True):
        current_time = row['date']
        if prev_time is None:
            session_start = current_time
            session_end = current_time
            session_count = 1
        else:
            gap = (current_time - prev_time).total_seconds() / 60
            if gap > gap_minutes:
                duration = session_end - session_start
                if duration > best_duration and session_count >= 3:
                    best_duration = duration
                    best_start = session_start
                    best_end = session_end
                    best_count = session_count
                session_start = current_time
                session_end = current_time
                session_count = 1
            else:
                session_end = current_time
                session_count += 1
        prev_time = current_time

    if session_start is not None:
        duration = session_end - session_start
        if duration > best_duration and session_count >= 3:
            best_duration = duration
            best_start = session_start
            best_end = session_end
            best_count = session_count

    return {
        'start': best_start,
        'end': best_end,
        'duration': best_duration,
        'messages': best_count
    }

def _night_activity(df):
    total = df.height
    night_hours = [23, 0, 1, 2, 3, 4]
    night_df = df.filter(pl.col('hour').is_in(night_hours))
    night_count = night_df.height
    night_pct = night_count / total * 100 if total else 0

    if night_df.height > 0:
        hourly_night = night_df.group_by('hour').agg(pl.count().alias('count')).sort('count', descending=True)
        peak_night_hour = int(hourly_night['hour'][0])
        peak_night_count = int(hourly_night['count'][0])
    else:
        peak_night_hour = 0
        peak_night_count = 0

    return {
        'total': total,
        'night_count': night_count,
        'night_pct': night_pct,
        'peak_hour': peak_night_hour,
        'peak_count': peak_night_count
    }

def _time_spent(df):
    date_min = df['date'].min()
    date_max = df['date'].max()
    span = date_max - date_min

    sessions = analyze_sessions(df, gap_hours=8)
    total_session_hours = float(sessions['duration_hours'].sum()) if sessions.height > 0 else 0

    return {
        'span': span,
        'total_session_hours': total_session_hours,
        'first_msg': date_min,
        'last_msg': date_max
    }

def _creative_comparisons(hours):
    comparisons = []
    movies = int(hours // 2)
    if movies > 0:
        comparisons.append(f"посмотреть {movies} {_pluralize(movies, 'фильм', 'фильма', 'фильмов')}")
    books = int(hours // 10)
    if books > 0:
        comparisons.append(f"прочитать {books} {_pluralize(books, 'книгу', 'книги', 'книг')}")
    km = int(hours * 5)
    if km > 0:
        comparisons.append(f"пройти {km} {_pluralize(km, 'километр', 'километра', 'километров')} пешком")
    earth_rounds = int((hours * 900) // 40000)
    if earth_rounds > 0:
        comparisons.append(f"облететь Землю {earth_rounds} {_pluralize(earth_rounds, 'раз', 'раза', 'раз')}")
    sleep_days = int(hours // 8)
    if sleep_days > 0:
        comparisons.append(f"поспать {sleep_days} {_pluralize(sleep_days, 'день', 'дня', 'дней')}")
    series = int(hours * 60 // 45)
    if series > 0:
        comparisons.append(f"посмотреть {series} {_pluralize(series, 'серию', 'серии', 'серий')} сериала")
    return comparisons[:4]

# ---------- ГЕНЕРАЦИЯ СЛАЙДОВ ----------
def generate_wrapped_slides(df, gift_df, cfg):
    W, H = 1080, 1920
    bg1 = _hex_to_rgb('#0a0a1a')
    bg2 = _hex_to_rgb('#1a0a2e')
    text_main = (255, 255, 255)
    text_muted = (180, 180, 200)
    accent1 = _hex_to_rgb(cfg['colors']['user1'])
    accent2 = _hex_to_rgb(cfg['colors']['user2'])

    senders = df['sender'].unique().to_list()
    if len(senders) < 2:
        senders = senders + ['—'] * (2 - len(senders))
    u1, u2 = senders[0], senders[1]
    u1_clean = _clean_name(u1)
    u2_clean = _clean_name(u2)

    total_msgs = df.height
    total_reactions = int(df['reactions_count'].sum())
    total_gifts = int(gift_df['gifts'].sum()) if gift_df is not None and gift_df.height > 0 else 0
    total_replies = int(df['reply_to'].is_not_null().sum())

    m1 = df.filter(pl.col('sender') == u1).height
    m2 = df.filter(pl.col('sender') == u2).height if u2 != '—' else 0

    date_min = df['date'].min().date()
    date_max = df['date'].max().date()
    period_str = f"{date_min.strftime('%d.%m.%Y')} — {date_max.strftime('%d.%m.%Y')}"

    def top_words(sender, n=5):
        stop = set(cfg['analysis']['stop_words'])
        stop.add('это')
        texts = df.filter((pl.col('sender') == sender) & (pl.col('media_type') == 'text'))['text'].to_list()
        c = Counter()
        for t in texts:
            t = t.lower()
            t = re.sub(r'[^\w\s]', '', t)
            c.update(w for w in t.split() if w not in stop and len(w) >= 3)
        return c.most_common(n)

    tw1_list = top_words(u1, 5)
    tw2_list = top_words(u2, 5)

    hourly = df.group_by('hour').agg(pl.count().alias('count')).sort('count', descending=True)
    peak_hour = int(hourly['hour'][0]) if hourly.height > 0 else 0
    peak_count = int(hourly['count'][0]) if hourly.height > 0 else 0

    days_ru = {'Monday':'ПОНЕДЕЛЬНИК','Tuesday':'ВТОРНИК','Wednesday':'СРЕДА',
               'Thursday':'ЧЕТВЕРГ','Friday':'ПЯТНИЦА','Saturday':'СУББОТА','Sunday':'ВОСКРЕСЕНЬЕ'}
    weekday = df.group_by('weekday_name').agg(pl.count().alias('count')).sort('count', descending=True)
    peak_day = days_ru.get(weekday['weekday_name'][0], '—') if weekday.height > 0 else '—'

    if weekday.height > 0:
        peak_weekday_en = weekday['weekday_name'][0]
        total_msgs_on_peak = int(weekday.filter(pl.col('weekday_name') == peak_weekday_en)['count'][0])
        occurrences = df.filter(pl.col('weekday_name') == peak_weekday_en)['day'].n_unique()
        peak_day_avg = total_msgs_on_peak // max(1, occurrences)
        peak_day_pct = total_msgs_on_peak / total_msgs * 100 if total_msgs else 0
        if weekday.height > 1:
            second_day_en = weekday['weekday_name'][1]
            second_msgs = int(weekday.filter(pl.col('weekday_name') == second_day_en)['count'][0])
            second_day_ru = days_ru.get(second_day_en, '—')
            diff_pct = ((total_msgs_on_peak - second_msgs) / max(1, second_msgs)) * 100
        else:
            second_day_ru = None
            diff_pct = 0
    else:
        peak_day_avg = 0
        peak_day_pct = 0
        second_day_ru = None
        diff_pct = 0

    longest = _find_longest_dialog(df, gap_minutes=30)
    night = _night_activity(df)
    time_info = _time_spent(df)

    slides = []

    # ============ СЛАЙД 1: ТИТУЛЬНЫЙ ============
    img = _gradient_bg((W, H), bg1, bg2)
    draw = ImageDraw.Draw(img)
    _draw_orb(img, 200, 300, 180, accent1, alpha=60)
    _draw_orb(img, 900, 1500, 220, accent2, alpha=60)
    _draw_orb(img, 850, 400, 120, (255,255,255), alpha=40)

    font_title = _load_font(70, bold=True)
    font_huge = _load_font(240, bold=True)
    font_sub = _load_font(50)
    font_period = _load_font(42)

    names_text = f'{u1_clean} × {u2_clean}'
    _draw_glow_text(img, (W//2, 500), names_text, font_title, text_main, accent1, glow_radius=15)

    _draw_glow_text(img, (W//2, 800), _format_number(total_msgs), font_huge, text_main, accent1, glow_radius=15)
    draw.text((W//2, 1000), 'сообщений', font=font_sub, fill=text_muted, anchor='mm')

    draw.line([(W//2 - 200, 1150), (W//2 + 200, 1150)], fill=accent1, width=3)

    draw.text((W//2, 1280), period_str, font=font_period, fill=text_muted, anchor='mm')

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    slides.append(buf)

    # ============ СЛАЙД 2: КТО ПИШЕТ БОЛЬШЕ ============
    img = _gradient_bg((W, H), bg1, bg2)
    draw = ImageDraw.Draw(img)
    _draw_orb(img, 150, 1600, 200, accent1, alpha=50)
    _draw_orb(img, 950, 400, 180, accent2, alpha=50)

    font_section = _load_font(60, bold=True)
    font_name = _load_font(55, bold=True)
    font_num = _load_font(90, bold=True)
    font_pct = _load_font(45)

    draw.text((W//2, 280), 'КТО ПИШЕТ БОЛЬШЕ', font=font_section, fill=text_muted, anchor='mm')

    total = m1 + m2
    pct1 = m1 / total * 100 if total else 0
    pct2 = m2 / total * 100 if total else 0

    bar_h = 180
    bar_max_w = W - 200
    y1 = 600
    y2 = 950

    draw.text((100, y1), u1_clean, font=font_name, fill=accent1, anchor='lm')
    w1 = int(bar_max_w * pct1 / 100)
    _draw_rounded_rect(draw, (100, y1 + 80, 100 + w1, y1 + 80 + bar_h), radius=20, fill=accent1)
    _draw_glow_text(img, (100 + w1 // 2, y1 + 80 + bar_h // 2), _format_number(m1), font_num, (0,0,0), accent1, glow_radius=0)
    draw.text((W - 100, y1 + 80 + bar_h // 2), f'{pct1:.0f}%', font=font_pct, fill=accent1, anchor='rm')

    draw.text((100, y2), u2_clean, font=font_name, fill=accent2, anchor='lm')
    w2 = int(bar_max_w * pct2 / 100)
    _draw_rounded_rect(draw, (100, y2 + 80, 100 + w2, y2 + 80 + bar_h), radius=20, fill=accent2)
    _draw_glow_text(img, (100 + w2 // 2, y2 + 80 + bar_h // 2), _format_number(m2), font_num, (0,0,0), accent2, glow_radius=0)
    draw.text((W - 100, y2 + 80 + bar_h // 2), f'{pct2:.0f}%', font=font_pct, fill=accent2, anchor='rm')

    winner = u1_clean if m1 >= m2 else u2_clean
    diff = abs(m1 - m2) / max(m1, m2) * 100 if max(m1, m2) else 0
    font_insight = _load_font(48)
    draw.text((W//2, 1500), f'{winner} пишет активнее на {diff:.0f}%', font=font_insight, fill=text_muted, anchor='mm')

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    slides.append(buf)

    # ============ СЛАЙД 3: ТОП-5 СЛОВ ОБОИХ ============
    img = _gradient_bg((W, H), bg1, bg2)
    draw = ImageDraw.Draw(img)
    _draw_orb(img, 100, 200, 200, accent1, alpha=40)
    _draw_orb(img, 980, 1700, 220, accent2, alpha=40)

    font_section = _load_font(55, bold=True)
    font_word = _load_font(70, bold=True)
    font_count = _load_font(45, bold=True)
    font_name = _load_font(48, bold=True)

    draw.text((W//2, 200), 'ГЛАВНЫЕ СЛОВА', font=font_section, fill=text_muted, anchor='mm')

    draw.line([(W//2, 280), (W//2, H - 200)], fill=(60, 60, 80), width=2)

    draw.text((W//4, 340), u1_clean.upper(), font=font_name, fill=accent1, anchor='mm')
    for i, (word, count) in enumerate(tw1_list):
        y = 450 + i * 180
        _draw_glow_text(img, (W//4, y), word.upper(), font_word, text_main, accent1, glow_radius=12)
        draw.text((W//4, y + 70), f'{_format_number(count)} {_pluralize(count, "раз", "раза", "раз")}', font=font_count, fill=text_muted, anchor='mm')

    draw.text((3*W//4, 340), u2_clean.upper(), font=font_name, fill=accent2, anchor='mm')
    for i, (word, count) in enumerate(tw2_list):
        y = 450 + i * 180
        _draw_glow_text(img, (3*W//4, y), word.upper(), font_word, text_main, accent2, glow_radius=12)
        draw.text((3*W//4, y + 70), f'{_format_number(count)} {_pluralize(count, "раз", "раза", "раз")}', font=font_count, fill=text_muted, anchor='mm')

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    slides.append(buf)

    # ============ СЛАЙД 4: ЗАТРАЧЕННОЕ ВРЕМЯ ============
    img = _gradient_bg((W, H), bg1, bg2)
    draw = ImageDraw.Draw(img)
    _draw_orb(img, 900, 400, 250, accent1, alpha=50)
    _draw_orb(img, 150, 1500, 200, accent2, alpha=50)

    font_section = _load_font(55, bold=True)
    font_huge = _load_font(140, bold=True)
    font_sub = _load_font(45)
    font_item = _load_font(42)

    draw.text((W//2, 250), 'ВРЕМЯ В ЧАТЕ', font=font_section, fill=text_muted, anchor='mm')

    total_hours = time_info['total_session_hours']
    hours_int = int(total_hours)
    _draw_glow_text(img, (W//2, 500), _format_number(hours_int), font_huge, text_main, accent1, glow_radius=15)
    draw.text((W//2, 620), _pluralize(hours_int, 'час', 'часа', 'часов') + ' активного общения', font=font_sub, fill=text_muted, anchor='mm')

    span_str = _format_timedelta(time_info['span'])
    draw.text((W//2, 750), f'За период: {span_str}', font=font_sub, fill=text_muted, anchor='mm')

    comparisons = _creative_comparisons(total_hours)
    draw.text((W//2, 900), 'ЗА ЭТО ВРЕМЯ МОЖНО БЫЛО:', font=font_sub, fill=accent2, anchor='mm')

    for i, comp in enumerate(comparisons):
        y = 1050 + i * 130
        draw.text((W//2, y), comp, font=font_item, fill=text_main, anchor='mm')

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    slides.append(buf)

    # ============ СЛАЙД 5: ПИК АКТИВНОСТИ ============
    img = _gradient_bg((W, H), bg1, bg2)
    draw = ImageDraw.Draw(img)
    _draw_orb(img, 900, 500, 220, accent1, alpha=50)
    _draw_orb(img, 150, 1400, 180, accent2, alpha=50)

    draw.text((W//2, 280), 'ВЫ ЧАЩЕ ВСЕГО ПИШЕТЕ В', font=font_section, fill=text_muted, anchor='mm')

    font_time = _load_font(260, bold=True)
    _draw_glow_text(img, (W//2, 700), f'{peak_hour:02d}:00', font_time, text_main, accent1, glow_radius=20)

    font_count_small = _load_font(50, bold=True)
    font_msg_text = _load_font(42)
    draw.text((W//2, 900), f'{_format_number(peak_count)} {_pluralize(peak_count, "сообщение", "сообщения", "сообщений")}', font=font_count_small, fill=text_main, anchor='mm')
    draw.text((W//2, 980), 'в этот час', font=font_msg_text, fill=text_muted, anchor='mm')

    hourly_sorted = df.group_by('hour').agg(pl.count().alias('count')).sort('hour').to_pandas()
    hourly_sorted = hourly_sorted.set_index('hour').reindex(range(24)).fillna(0)
    counts = hourly_sorted['count'].tolist()
    max_c = max(counts) if counts else 1
    graph_w = W - 200
    graph_h = 300
    graph_x = 100
    graph_y = 1100

    points = []
    for i, c in enumerate(counts):
        x = graph_x + (i / 23) * graph_w
        y = graph_y + graph_h - (c / max_c) * graph_h
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=accent1, width=6)
        peak_idx = counts.index(max_c)
        px_p, py_p = points[peak_idx]
        draw.ellipse((px_p-15, py_p-15, px_p+15, py_p+15), fill=accent1)

    font_small = _load_font(32)
    for h in [0, 6, 12, 18, 23]:
        x = graph_x + (h / 23) * graph_w
        draw.text((x, graph_y + graph_h + 20), f'{h:02d}', font=font_small, fill=text_muted, anchor='mm')

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    slides.append(buf)

    # ============ СЛАЙД 6: АКТИВНЫЙ ДЕНЬ ============
    img = _gradient_bg((W, H), bg1, bg2)
    draw = ImageDraw.Draw(img)
    _draw_orb(img, 540, 400, 300, accent2, alpha=40)

    draw.text((W//2, 250), 'САМЫЙ АКТИВНЫЙ ДЕНЬ', font=font_section, fill=text_muted, anchor='mm')

    font_day = _load_font(140, bold=True)
    _draw_glow_text(img, (W//2, 650), peak_day, font_day, text_main, accent2, glow_radius=18)

    font_count_day = _load_font(55, bold=True)
    draw.text((W//2, 900), f'~{_format_number(peak_day_avg)} {_pluralize(peak_day_avg, "сообщение", "сообщения", "сообщений")}', font=font_count_day, fill=text_main, anchor='mm')
    draw.text((W//2, 980), 'в этот день в среднем', font=font_sub, fill=text_muted, anchor='mm')

    font_pct_big = _load_font(100, bold=True)
    _draw_glow_text(img, (W//2, 1180), f'{peak_day_pct:.1f}%', font_pct_big, accent2, accent2, glow_radius=10)
    draw.text((W//2, 1280), 'всех сообщений чата', font=font_sub, fill=text_muted, anchor='mm')

    if second_day_ru and diff_pct > 0:
        font_cmp = _load_font(40)
        draw.text((W//2, 1380), f'на {diff_pct:.0f}% больше, чем {second_day_ru}', font=font_cmp, fill=text_muted, anchor='mm')

    days_short = ['ПН','ВТ','СР','ЧТ','ПТ','СБ','ВС']
    days_full = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    circle_y = 1550
    circle_r = 45
    spacing = 110
    start_x = W//2 - 3 * spacing
    for i, (short, full) in enumerate(zip(days_short, days_full)):
        cx = start_x + i * spacing
        is_peak = (full == weekday['weekday_name'][0]) if weekday.height > 0 else False
        color = accent2 if is_peak else (60, 60, 80)
        draw.ellipse((cx-circle_r, circle_y-circle_r, cx+circle_r, circle_y+circle_r), fill=color)
        draw.text((cx, circle_y), short, font=_load_font(28, bold=True), fill=(0,0,0) if is_peak else text_muted, anchor='mm')

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    slides.append(buf)

    # ============ СЛАЙД 7: САМЫЙ ДЛИННЫЙ ДИАЛОГ ============
    img = _gradient_bg((W, H), bg1, bg2)
    draw = ImageDraw.Draw(img)
    _draw_orb(img, 200, 400, 220, accent1, alpha=50)
    _draw_orb(img, 880, 1500, 250, accent2, alpha=50)

    draw.text((W//2, 250), 'САМЫЙ ДЛИННЫЙ ДИАЛОГ', font=font_section, fill=text_muted, anchor='mm')

    font_dur = _load_font(110, bold=True)
    duration_str = _format_timedelta(longest['duration'])
    _draw_glow_text(img, (W//2, 550), duration_str, font_dur, text_main, accent1, glow_radius=15)

    font_msg_count = _load_font(50, bold=True)
    msg_count = longest['messages']
    draw.text((W//2, 720), f'{msg_count} {_pluralize(msg_count, "сообщение", "сообщения", "сообщений")}', font=font_msg_count, fill=text_main, anchor='mm')

    font_dt = _load_font(42)
    if longest['start'] and longest['end']:
        start_str = longest['start'].strftime('%d.%m.%Y %H:%M')
        end_str = longest['end'].strftime('%H:%M')
        draw.text((W//2, 900), f'Начало: {start_str}', font=font_dt, fill=text_muted, anchor='mm')
        draw.text((W//2, 980), f'Конец: {end_str}', font=font_dt, fill=text_muted, anchor='mm')

    draw.line([(W//2 - 250, 1100), (W//2 + 250, 1100)], fill=accent2, width=2)

    font_fact = _load_font(38)
    draw.text((W//2, 1200), 'Непрерывный обмен', font=font_fact, fill=text_muted, anchor='mm')
    draw.text((W//2, 1270), 'без пауз более 30 минут', font=font_fact, fill=text_muted, anchor='mm')

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    slides.append(buf)

    # ============ СЛАЙД 8: НОЧНАЯ АКТИВНОСТЬ ============
    img = _gradient_bg((W, H), bg1, bg2)
    draw = ImageDraw.Draw(img)
    _draw_orb(img, 540, 300, 280, (100, 100, 200), alpha=40)
    _draw_orb(img, 200, 1500, 180, accent1, alpha=40)

    draw.text((W//2, 200), 'НОЧНАЯ АКТИВНОСТЬ', font=font_section, fill=text_muted, anchor='mm')
    draw.text((W//2, 280), '23:00 — 05:00', font=font_sub, fill=text_muted, anchor='mm')

    font_pct = _load_font(180, bold=True)
    _draw_glow_text(img, (W//2, 550), f'{night["night_pct"]:.1f}%', font_pct, text_main, accent1, glow_radius=20)
    draw.text((W//2, 700), 'сообщений ночью', font=font_sub, fill=text_muted, anchor='mm')

    font_num = _load_font(70, bold=True)
    nc = night['night_count']
    draw.text((W//4, 900), _format_number(nc), font=font_num, fill=accent1, anchor='mm')
    draw.text((W//4, 980), 'ночных', font=font_sub, fill=text_muted, anchor='mm')

    tc = night['total']
    draw.text((3*W//4, 900), _format_number(tc), font=font_num, fill=accent2, anchor='mm')
    draw.text((3*W//4, 980), 'всего', font=font_sub, fill=text_muted, anchor='mm')

    font_peak = _load_font(100, bold=True)
    _draw_glow_text(img, (W//2, 1250), f'{night["peak_hour"]:02d}:00', font_peak, text_main, accent2, glow_radius=15)
    draw.text((W//2, 1350), 'самый активный ночной час', font=font_sub, fill=text_muted, anchor='mm')
    npc = night['peak_count']
    draw.text((W//2, 1430), f'{_format_number(npc)} {_pluralize(npc, "сообщение", "сообщения", "сообщений")}', font=font_count_day, fill=text_main, anchor='mm')

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    slides.append(buf)

    # ============ СЛАЙД 9: ФИНАЛЬНЫЙ ============
    img = _gradient_bg((W, H), bg1, bg2)
    draw = ImageDraw.Draw(img)
    _draw_orb(img, 200, 400, 200, accent1, alpha=60)
    _draw_orb(img, 850, 1500, 250, accent2, alpha=60)
    _draw_orb(img, 900, 300, 150, (255,255,255), alpha=40)
    _draw_orb(img, 150, 1600, 180, (255,255,255), alpha=30)

    font_final = _load_font(130, bold=True)
    font_date = _load_font(50)

    final_names = f'{u1_clean} × {u2_clean}'
    _draw_glow_text(img, (W//2, 700), final_names, font_final, text_main, accent1, glow_radius=20)

    draw.line([(W//2 - 250, 850), (W//2 + 250, 850)], fill=accent2, width=3)

    draw.text((W//2, 1000), f'{_format_number(total_msgs)} {_pluralize(total_msgs, "сообщение", "сообщения", "сообщений")}', font=font_count_day, fill=text_main, anchor='mm')
    draw.text((W//2, 1100), f'{_format_number(total_reactions)} {_pluralize(total_reactions, "реакция", "реакции", "реакций")}', font=font_count_day, fill=text_main, anchor='mm')
    draw.text((W//2, 1200), f'{_format_number(total_gifts)} {_pluralize(total_gifts, "подарок", "подарка", "подарков")}', font=font_count_day, fill=text_main, anchor='mm')
    draw.text((W//2, 1300), f'{_format_number(total_replies)} {_pluralize(total_replies, "ответ", "ответа", "ответов")}', font=font_count_day, fill=text_main, anchor='mm')

    draw.text((W//2, 1700), f'Сгенерировано {datetime.now().strftime("%d.%m.%Y")}', font=font_date, fill=text_muted, anchor='mm')

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    slides.append(buf)

    return slides

# ---------- ЗАГРУЗКА ФАЙЛА ----------
uploaded_file = st.sidebar.file_uploader(
    "Загрузите JSON-экспорт чата",
    type=["json"],
    key="json_uploader"
)

def get_file_hash(file):
    return hashlib.md5(file.getvalue()).hexdigest()

if uploaded_file is not None:
    file_hash = get_file_hash(uploaded_file)
    if 'file_hash' not in st.session_state or st.session_state.file_hash != file_hash:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        with st.spinner("Загрузка и парсинг данных..."):
            raw = load_data(tmp_path)
            df = parse_messages(raw)
            gift_days, gift_counts = gifts_by_day(raw)
            if gift_days.len() > 0:
                gift_df = pl.DataFrame({'day': gift_days, 'gifts': gift_counts})
            else:
                gift_df = pl.DataFrame(schema={'day': pl.Date, 'gifts': pl.Int32})
            service_days, service_counts = service_by_day(raw)
            if service_days.len() > 0:
                service_df = pl.DataFrame({'day': service_days, 'service_count': service_counts})
            else:
                service_df = pl.DataFrame(schema={'day': pl.Date, 'service_count': pl.Int32})
            call_stats = call_duration_stats(raw)
            os.unlink(tmp_path)
        st.session_state.df = df
        st.session_state.raw = raw
        st.session_state.gift_df = gift_df
        st.session_state.service_df = service_df
        st.session_state.call_stats = call_stats
        st.session_state.file_hash = file_hash
        st.session_state.file_loaded = True
    else:
        df = st.session_state.df
        raw = st.session_state.raw
        gift_df = st.session_state.gift_df
        service_df = st.session_state.service_df
        call_stats = st.session_state.call_stats

    if df.height == 0:
        st.error("Не удалось распарсить файл.")
    else:
        st.success(f"Загружено {df.height} обычных сообщений из {len(raw)} записей")
        st.sidebar.subheader("Фильтры")
        date_min = df['date'].min().date()
        date_max = df['date'].max().date()
        date_range = st.sidebar.date_input("Период", [date_min, date_max])
        if len(date_range) == 2:
            start_date, end_date = date_range
            mask = (pl.col('date').cast(pl.Date) >= start_date) & (pl.col('date').cast(pl.Date) <= end_date)
            df_filtered = df.filter(mask)
        else:
            df_filtered = df
        senders = df_filtered['sender'].unique().to_list()
        selected_users = st.sidebar.multiselect("Выберите пользователей", senders, default=senders)
        if selected_users:
            df_filtered = df_filtered.filter(pl.col('sender').is_in(selected_users))
        max_rows = st.sidebar.number_input("Максимум сообщений для анализа (0 – все)",
                                           min_value=0, max_value=1000000, value=100000, step=10000)
        if max_rows > 0 and df_filtered.height > max_rows:
            df_filtered = df_filtered.tail(max_rows)
            st.sidebar.warning(f"Анализ ограничен {max_rows} последними сообщениями")
        top_n = st.sidebar.slider("Количество топ-слов", 5, 50, top_n_default)
        days_ahead = st.sidebar.slider("Прогноз на дней вперёд", 1, 30, 7)
        include_service = st.sidebar.checkbox("Включить сервисные записи (звонки, подарки, закрепления)", value=False)

        st.sidebar.markdown("---")

        # ---------- ЭКСПОРТ В CSV ----------
        def _export_to_csv(sub_df):
            lines = []
            for row in sub_df.sort('date').iter_rows(named=True):
                dt = row['date'].strftime('%Y-%m-%d %H:%M:%S')
                sender = row['sender']
                reply_to = row.get('reply_to', '') or ''
                text = row.get('text', '') or ''
                lines.append(f"{dt}|{sender}|{reply_to}|{text}")
            return '\n'.join(lines)

        if st.sidebar.button("Экспорт в CSV", use_container_width=True):
            csv_content = _export_to_csv(df_filtered)
            st.sidebar.download_button(
                label="Скачать CSV",
                data=csv_content,
                file_name="chat_export.csv",
                mime="text/csv"
            )

        if st.sidebar.button("Создать «Итоги чата»", type="primary", use_container_width=True):
            if df_filtered['sender'].n_unique() < 2:
                st.sidebar.warning("Нужно минимум 2 пользователя")
            else:
                with st.spinner("Генерация слайдов..."):
                    slides = generate_wrapped_slides(df_filtered, gift_df, config)
                    st.sidebar.markdown(f"**Сгенерировано слайдов: {len(slides)}**")
                    for i, buf in enumerate(slides, 1):
                        st.sidebar.image(buf, caption=f"Слайд {i}", use_container_width=True)
                        buf.seek(0)
                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for i, buf in enumerate(slides, 1):
                            buf.seek(0)
                            zf.writestr(f"slide_{i:02d}.png", buf.read())
                    zip_buf.seek(0)
                    st.sidebar.download_button(
                        label="Скачать все слайды (ZIP)",
                        data=zip_buf,
                        file_name="chat_wrapped.zip",
                        mime="application/zip"
                    )

        @st.cache_data(ttl=3600, show_spinner=False)
        def cached_daily_analytics(df, gift_df, service_df, include_service):
            return daily_analytics(df, gift_df, service_df, include_service)
        @st.cache_data(ttl=3600, show_spinner=False)
        def cached_monthly_analytics(df, gift_df):
            return monthly_analytics(df, gift_df)
        @st.cache_data(ttl=3600, show_spinner=False)
        def cached_frequent_words(df, stop_words_tuple, top_n, min_len):
            return analyze_frequent_words(df, set(stop_words_tuple), top_n, min_len)
        @st.cache_data(ttl=3600, show_spinner=False)
        def cached_bigrams(df, stop_words_tuple, min_len, top_bigrams):
            return analyze_bigrams(df, set(stop_words_tuple), min_len, top_bigrams)
        @st.cache_data(ttl=3600, show_spinner=False)
        def cached_reactions(df):
            return reactions_breakdown(df)
        @st.cache_data(ttl=3600, show_spinner=False)
        def cached_participant_stats(df):
            return participant_stats(df)
        @st.cache_data(ttl=3600, show_spinner=False)
        def cached_gap_analysis(df, max_gap):
            return reply_gap_by_sender(df, max_gap)
        @st.cache_data(ttl=3600, show_spinner=False)
        def cached_sessions(df, gap):
            return analyze_sessions(df, gap)
        @st.cache_data(ttl=3600, show_spinner=False)
        def cached_predict(df, days):
            return predict_activity(df, days)
        @st.cache_data(ttl=3600, show_spinner=False)
        def cached_general_stats(df, gifts_total):
            return general_stats(df, gifts_total)

        tabs = st.tabs([
            "Общая статистика",
            "Качество данных",
            "Участники",
            "Реакции",
            "Активность",
            "Частотные слова",
            "Биграммы",
            "Сессии",
            "Длина текстов",
            "Темп диалога",
            "Ежедневно",
            "Месячная статистика",
            "Сервисные и редактирования",
            "Сравнение",
            "Прогноз"
        ])

        with tabs[0]:
            gifts_total = gift_df['gifts'].sum() if gift_df.height > 0 else 0
            stats = cached_general_stats(df_filtered, gifts_total)
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Сообщений", stats['total_messages'])
            col2.metric("Ответов", stats['replies'])
            col3.metric("Реакций", stats['total_reactions'])
            col4.metric("Подарков", stats['gifts'])
            col5.metric("Редактирований", stats['edited'])
            st.subheader("Сообщения по пользователям")
            sender_df = pd.DataFrame(list(stats['sender_counts'].items()), columns=['Пользователь', 'Количество'])
            fig = px.pie(sender_df, values='Количество', names='Пользователь',
                         color_discrete_sequence=[colors['user1'], colors['user2']])
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Типы сообщений")
            media_df = pd.DataFrame(list(stats['media_counts'].items()), columns=['Тип', 'Количество'])
            media_df = media_df.sort_values('Количество', ascending=False)
            fig = px.bar(media_df, x='Тип', y='Количество', color='Тип')
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Средняя длина текста", f"{stats['avg_text_length']:.1f} символов")

        with tabs[1]:
            quality = data_quality_check(df_filtered, raw)
            st.subheader("Контроль качества")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Всего записей", quality['total_records'])
            col2.metric("Уникальных id", quality['unique_ids'])
            col3.metric("Дубликатов", quality['duplicates'])
            col4.metric("Распознанных дат", f"{quality['dates_parsed']} / {len(raw)}")
            st.write(f"Порядок времени: {'Корректен' if quality['ordered'] else 'Нарушен'}")

        with tabs[2]:
            part_stats = cached_participant_stats(df_filtered)
            if part_stats:
                st.subheader("Участники и объём участия")
                df_ps = pd.DataFrame(part_stats).T.reset_index().rename(columns={'index': 'sender'})
                st.dataframe(df_ps)
            else:
                st.info("Нет данных")

        with tabs[3]:
            reactions = cached_reactions(df_filtered)
            if reactions:
                st.subheader("Реакции по типам")
                react_df = pd.DataFrame(list(reactions.items()), columns=['Реакция', 'Количество'])
                react_df = react_df.sort_values('Количество', ascending=True)
                fig = px.bar(react_df, x='Реакция', y='Количество', color='Реакция', title='События реакций')
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(react_df)
            else:
                st.info("Нет реакций")

        with tabs[4]:
            st.subheader("Активность по часам")
            hourly = df_filtered.group_by(['hour', 'sender']).agg(pl.count().alias('count')).sort('hour').to_pandas()
            fig = px.line(hourly, x='hour', y='count', color='sender',
                          title='Сообщения по часам', markers=True,
                          color_discrete_sequence=[colors['user1'], colors['user2']])
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Тепловая карта: день недели / час")
            pivot = df_filtered.group_by(['weekday_name', 'hour']).agg(pl.count().alias('count')).to_pandas()
            pivot = pivot.pivot(index='weekday_name', columns='hour', values='count').fillna(0)
            days_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            pivot = pivot.reindex(days_order, axis=0)
            fig = px.imshow(pivot, labels=dict(x="Час", y="День недели", color="Сообщений"),
                            color_continuous_scale=config['plots']['heatmap'], aspect="auto")
            st.plotly_chart(fig, use_container_width=True)

        with tabs[5]:
            freq_result = cached_frequent_words(df_filtered, tuple(stop_words), top_n, min_len)
            if freq_result:
                st.subheader(f"Общий топ-{top_n}")
                overall_df = pd.DataFrame(freq_result['overall'], columns=['Слово', 'Частота'])
                fig = px.bar(overall_df, x='Слово', y='Частота', color='Слово',
                             title=f'Самые частые слова (топ-{top_n})')
                st.plotly_chart(fig, use_container_width=True)
                st.subheader("По пользователям")
                for sender, top_words in freq_result['users'].items():
                    with st.expander(f"{sender}"):
                        if top_words:
                            df_sender = pd.DataFrame(top_words, columns=['Слово', 'Частота'])
                            fig = px.bar(df_sender, x='Слово', y='Частота', color='Слово',
                                         title=f'Топ-{len(top_words)} слов для {sender}')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.write("Нет слов.")
            else:
                st.info("Нет текстовых сообщений")

        with tabs[6]:
            bigrams = cached_bigrams(df_filtered, tuple(stop_words), min_len, top_bigrams)
            if bigrams:
                st.subheader(f"Топ-{len(bigrams)} биграмм")
                bigram_df = pd.DataFrame([(' '.join(b), c) for b, c in bigrams], columns=['Биграмма', 'Частота'])
                fig = px.bar(bigram_df, x='Биграмма', y='Частота', color='Биграмма',
                             title='Самые частые биграммы')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Нет данных для биграмм.")

        with tabs[7]:
            sessions_df = cached_sessions(df_filtered, session_gap)
            if sessions_df.height > 0:
                st.subheader("Статистика сессий")
                sessions_pd = sessions_df.to_pandas()
                col1, col2, col3 = st.columns(3)
                col1.metric("Всего сессий", sessions_pd.shape[0])
                col2.metric("Средняя длительность", f"{sessions_pd['duration_hours'].mean():.1f} ч")
                col3.metric("Медианная длительность", f"{sessions_pd['duration_hours'].median():.1f} ч")
                fig = px.histogram(sessions_pd, x='duration_hours', nbins=30,
                                   title='Распределение длительности сессий',
                                   labels={'duration_hours':'Часы'})
                st.plotly_chart(fig, use_container_width=True)
                st.subheader("Сессии по пользователям")
                sender_session = sessions_pd.groupby('sender').agg(
                    сессий=('duration_hours', 'size'),
                    средняя_длит=('duration_hours', 'mean'),
                    всего_сообщений=('messages', 'sum')
                ).reset_index()
                st.dataframe(sender_session)
            else:
                st.info("Нет данных для анализа сессий.")

        with tabs[8]:
            word_counts, char_counts = text_length_distribution(df_filtered)
            if word_counts.len() > 0:
                word_counts_pd = word_counts.to_pandas()
                st.subheader("Распределение слов в текстовых сообщениях")
                fig = px.histogram(word_counts_pd, nbins=50, title='Количество слов на сообщение')
                fig.add_vline(x=word_counts_pd.median(), line_dash="dash", line_color="red", annotation_text=f"Медиана: {word_counts_pd.median():.0f}")
                fig.add_vline(x=word_counts_pd.mean(), line_dash="dot", line_color="green", annotation_text=f"Среднее: {word_counts_pd.mean():.1f}")
                st.plotly_chart(fig, use_container_width=True)
                st.write(f"Медиана: {word_counts_pd.median():.0f} слов, 90-й перцентиль: {word_counts_pd.quantile(0.9):.0f}")
            else:
                st.info("Нет текстовых сообщений.")

        with tabs[9]:
            gap_results = cached_gap_analysis(df_filtered, config['analysis']['max_reply_gap_hours'])
            if gap_results:
                st.subheader("Интервалы при смене автора (по авторам)")
                gap_df = pd.DataFrame(gap_results).T.reset_index().rename(columns={'index': 'sender'})
                st.dataframe(gap_df)
                def all_gaps_series(df, max_hours):
                    df_sorted = df.sort('date')
                    gaps_list = []
                    prev_row = None
                    for row in df_sorted.iter_rows(named=True):
                        if prev_row is not None and row['sender'] != prev_row['sender']:
                            gap = (row['date'] - prev_row['date']).total_seconds() / 60
                            if gap <= max_hours * 60:
                                gaps_list.append(gap)
                        prev_row = row
                    return pl.Series(gaps_list)
                gaps_series = all_gaps_series(df_filtered, config['analysis']['max_reply_gap_hours'])
                if gaps_series.len() > 0:
                    gaps_pd = gaps_series.to_pandas()
                    fig = px.histogram(gaps_pd, nbins=30, title='Общее распределение интервалов (мин)')
                    fig.add_vline(x=gaps_pd.median(), line_dash="dash", line_color="red", annotation_text=f"Медиана: {gaps_pd.median():.2f}")
                    fig.add_vline(x=gaps_pd.mean(), line_dash="dot", line_color="green", annotation_text=f"Среднее: {gaps_pd.mean():.2f}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Недостаточно данных для анализа.")

        with tabs[10]:
            daily = cached_daily_analytics(df_filtered, gift_df, service_df, include_service)
            if daily.height > 0:
                daily_pd = daily.to_pandas()
                st.subheader("Ежедневная статистика (все дни)")
                st.dataframe(daily_pd)
                st.subheader("Динамика сообщений по дням")
                title = "Количество сообщений в день (включая сервисные)" if include_service else "Количество сообщений в день (только обычные)"
                fig = px.line(daily_pd, x='day', y='total', title=title)
                st.plotly_chart(fig, use_container_width=True)
                st.subheader("Накопленный рост")
                daily_pd['cumsum'] = daily_pd['total'].cumsum()
                fig = px.line(daily_pd, x='day', y='cumsum', title='Накопленное количество сообщений')
                st.plotly_chart(fig, use_container_width=True)

        with tabs[11]:
            monthly = cached_monthly_analytics(df_filtered, gift_df)
            if monthly.height > 0:
                monthly_pd = monthly.to_pandas()
                st.subheader("Месячная статистика")
                st.dataframe(monthly_pd)
                fig_messages = px.bar(monthly_pd, x='year_month',
                                      y=['messages', 'replies', 'media', 'gifts'],
                                      title='Сообщения, ответы, медиа и подарки по месяцам',
                                      barmode='group')
                st.plotly_chart(fig_messages, use_container_width=True)
                fig_words = px.line(monthly_pd, x='year_month',
                                    y=['messages', 'words', 'days_active'],
                                    title='Количество сообщений, слов и дней активности по месяцам',
                                    markers=True)
                st.plotly_chart(fig_words, use_container_width=True)
            else:
                st.info("Нет данных для месячной статистики.")

        with tabs[12]:
            service_info = service_actions_analysis(raw)
            gifts_total = gift_df['gifts'].sum() if gift_df.height > 0 else 0
            stats = cached_general_stats(df_filtered, gifts_total)
            st.subheader("Редактирования и сервисные записи")
            col1, col2 = st.columns(2)
            col1.metric("Отредактировано сообщений", f"{stats['edited']} ({stats['edited']/stats['total_messages']*100:.1f}%)")
            col2.metric("Сервисных записей", service_info['total_service'])
            if service_info['action_counts']:
                st.write("Типы действий:")
                st.dataframe(pd.DataFrame(service_info['action_counts'].items(), columns=['Действие', 'Количество']))
            if call_stats:
                st.subheader("Статистика звонков")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Всего звонков", call_stats['total_calls'])
                col2.metric("Общая длительность", f"{call_stats['total_duration_sec'] // 60} мин {call_stats['total_duration_sec'] % 60} сек")
                col3.metric("Средняя длительность", f"{call_stats['avg_duration_sec']:.0f} сек")
                col4.metric("Макс. длительность", f"{call_stats['max_duration']} сек")
                fig = px.histogram(call_stats['durations_list'], nbins=20,
                                   title='Распределение длительности звонков',
                                   labels={'value':'Длительность (сек)'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Нет данных о длительности звонков.")

        with tabs[13]:
            senders_list = df_filtered['sender'].unique().to_list()
            if len(senders_list) == 2:
                users = senders_list
                st.subheader(f"Сравнение: {users[0]} vs {users[1]}")
                hourly = df_filtered.group_by(['hour', 'sender']).agg(pl.count().alias('count')).sort('hour').to_pandas()
                fig = px.line(hourly, x='hour', y='count', color='sender',
                              title='Активность по часам',
                              color_discrete_sequence=[colors['user1'], colors['user2']])
                st.plotly_chart(fig, use_container_width=True)
                weekday = df_filtered.group_by(['weekday_name', 'sender']).agg(pl.count().alias('count')).sort('weekday_name').to_pandas()
                fig = px.bar(weekday, x='weekday_name', y='count', color='sender', barmode='group',
                             title='Активность по дням недели',
                             color_discrete_sequence=[colors['user1'], colors['user2']])
                st.plotly_chart(fig, use_container_width=True)
                media_by_user = df_filtered.group_by(['sender', 'media_type']).agg(pl.count().alias('count')).sort('count').to_pandas()
                media_by_user = media_by_user.sort_values('count', ascending=True)
                fig = px.bar(media_by_user, x='media_type', y='count', color='sender', barmode='group',
                             title='Типы сообщений по пользователям (по возрастанию)',
                             color_discrete_sequence=[colors['user1'], colors['user2']],
                             category_orders={"media_type": media_by_user['media_type'].tolist()})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Для сравнения нужно выбрать ровно двух пользователей.")

        with tabs[14]:
            pred_df, slope, intercept, daily_hist = cached_predict(df_filtered, days_ahead)
            if pred_df is not None and pred_df.height > 0:
                st.subheader(f"Прогноз активности на {days_ahead} дней")
                st.write(f"Тренд: {slope:.2f} сообщений в день")
                st.write(f"Базовый уровень: {intercept:.0f} сообщений в день")
                daily_hist_pd = daily_hist.to_pandas()
                pred_pd = pred_df.to_pandas()
                fig = px.line(daily_hist_pd, x='day', y='count', title='Ежедневная активность и прогноз')
                fig.add_scatter(x=pred_pd['date'], y=pred_pd['predicted'], mode='lines+markers',
                                name='Прогноз', line=dict(dash='dash', color='red'))
                fig.update_layout(xaxis_title='Дата', yaxis_title='Сообщений')
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(pred_pd)
            else:
                st.info("Недостаточно данных для прогноза.")

else:
    st.info("Загрузите JSON-файл экспорта чата Telegram для начала анализа.")

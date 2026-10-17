import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
from hydrox import *
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
import base64
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


# ---------- ВЁРСТКА СЛАЙДОВ: ПОЛЯ И ПОДГОНКА ТЕКСТА ----------
SLIDE_MX = 70  # боковые поля на всех слайдах

def _fit_font(draw, text, start_size, max_width, bold=False, min_size=22):
    """Уменьшает шрифт, пока текст не влезет в max_width."""
    text = str(text)
    size = start_size
    while size > min_size:
        font = _load_font(size, bold=bold)
        if draw.textlength(text, font=font) <= max_width:
            return font, size
        size -= 4
    return _load_font(min_size, bold=bold), min_size

def _wrap_lines(draw, text, font, max_width):
    """Переносит текст по словам под max_width."""
    words = str(text).split()
    if not words:
        return ['']
    lines, cur = [], ''
    for word in words:
        trial = f'{cur} {word}'.strip()
        if draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines

def _fit_wrapped(draw, text, start_size, max_width, bold=False, min_size=22, max_lines=3):
    """Подбирает шрифт и переносы: строк не больше max_lines, все влезают в max_width."""
    size = start_size
    while size > min_size:
        font = _load_font(size, bold=bold)
        lines = _wrap_lines(draw, text, font, max_width)
        if len(lines) <= max_lines and all(
            draw.textlength(l, font=font) <= max_width for l in lines
        ):
            return font, lines, size
        size -= 4
    # Страховка на минимальном размере: обрезаем не влезающее с многоточием
    font = _load_font(min_size, bold=bold)
    lines = _wrap_lines(draw, text, font, max_width)[:max_lines]
    fixed = []
    for line in lines:
        if draw.textlength(line, font=font) > max_width:
            while line and draw.textlength(line.rstrip() + '…', font=font) > max_width:
                line = line[:-1]
            line = line.rstrip() + '…'
        fixed.append(line)
    return font, fixed, min_size

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
        hourly_night = night_df.group_by('hour').agg(pl.len().alias('count')).sort('count', descending=True)
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
    MX = SLIDE_MX                    # единые боковые поля на всех слайдах
    CW = W - 2 * MX                  # ширина контентной области
    bg1 = _hex_to_rgb('#0a0a1a')
    bg2 = _hex_to_rgb('#1a0a2e')
    text_main = (255, 255, 255)
    text_muted = (180, 180, 200)
    divider = (60, 60, 80)
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

    hourly = df.group_by('hour').agg(pl.len().alias('count')).sort('count', descending=True)
    peak_hour = int(hourly['hour'][0]) if hourly.height > 0 else 0
    peak_count = int(hourly['count'][0]) if hourly.height > 0 else 0

    days_ru = {'Monday':'ПОНЕДЕЛЬНИК','Tuesday':'ВТОРНИК','Wednesday':'СРЕДА',
               'Thursday':'ЧЕТВЕРГ','Friday':'ПЯТНИЦА','Saturday':'СУББОТА','Sunday':'ВОСКРЕСЕНЬЕ'}
    weekday = df.group_by('weekday_name').agg(pl.len().alias('count')).sort('count', descending=True)
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

    # --- единые приёмы вёрстки ---
    def new_slide():
        img = _gradient_bg((W, H), bg1, bg2)
        return img, ImageDraw.Draw(img)

    def slide_title(draw, text):
        """Единый стиль заголовка на всех контентных слайдах."""
        font, _ = _fit_font(draw, text, 56, CW, bold=True)
        draw.text((W // 2, 225), text, font=font, fill=text_muted, anchor='mm')
        tw = draw.textlength(text, font=font)
        half = min(tw // 2 + 40, CW // 2)
        draw.line([(W // 2 - half, 275), (W // 2 + half, 275)], fill=divider, width=2)

    def centered_block(img, draw, y_center, lines, font, fill, glow=None, gap=1.3):
        """Рисует многострочный текст по центру без вылезания за края."""
        step = int(font.size * gap)
        y = y_center - (len(lines) - 1) * step // 2
        for line in lines:
            if glow is not None:
                _draw_glow_text(img, (W // 2, y), line, font, fill, glow,
                                glow_radius=max(6, font.size // 9))
            else:
                draw.text((W // 2, y), line, font=font, fill=fill, anchor='mm')
            y += step

    # ============ СЛАЙД 1: ТИТУЛЬНЫЙ ============
    img, draw = new_slide()
    _draw_orb(img, 200, 300, 180, accent1, alpha=60)
    _draw_orb(img, 900, 1500, 220, accent2, alpha=60)
    _draw_orb(img, 850, 400, 120, (255, 255, 255), alpha=40)

    names_text = f'{u1_clean} × {u2_clean}'
    nfont, nlines, _ = _fit_wrapped(draw, names_text, 72, CW, bold=True, max_lines=2)
    centered_block(img, draw, 480, nlines, nfont, text_main, glow=accent1)

    num_str = _format_number(total_msgs)
    fnum, _ = _fit_font(draw, num_str, 240, CW, bold=True)
    _draw_glow_text(img, (W // 2, 800), num_str, fnum, text_main, accent1, glow_radius=15)
    fsub, _ = _fit_font(draw, 'сообщений', 50, CW)
    draw.text((W // 2, 990), 'сообщений', font=fsub, fill=text_muted, anchor='mm')

    draw.line([(W // 2 - 200, 1120), (W // 2 + 200, 1120)], fill=accent1, width=3)

    fper, _ = _fit_font(draw, period_str, 42, CW)
    draw.text((W // 2, 1250), period_str, font=fper, fill=text_muted, anchor='mm')

    buf = io.BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
    slides.append(buf)

    # ============ СЛАЙД 2: КТО ПИШЕТ БОЛЬШЕ ============
    img, draw = new_slide()
    _draw_orb(img, 150, 1600, 200, accent1, alpha=50)
    _draw_orb(img, 950, 400, 180, accent2, alpha=50)
    slide_title(draw, 'КТО ПИШЕТ БОЛЬШЕ')

    total = m1 + m2
    pct1 = m1 / total * 100 if total else 0
    pct2 = m2 / total * 100 if total else 0

    bar_h = 150
    rows = [(u1_clean, m1, pct1, accent1, 560), (u2_clean, m2, pct2, accent2, 900)]
    for name, val, pct, color, y in rows:
        fname, _ = _fit_font(draw, name, 55, CW, bold=True)
        draw.text((MX, y), name, font=fname, fill=color, anchor='lm')
        bar_y = y + 70
        w = max(int(CW * pct / 100), 8)
        _draw_rounded_rect(draw, (MX, bar_y, MX + w, bar_y + bar_h), radius=20, fill=color)
        # число внутри, если влезает, иначе справа от полосы
        fval, _ = _fit_font(draw, _format_number(val), 80, max(w - 40, 100), bold=True)
        if draw.textlength(_format_number(val), font=fval) <= w - 40:
            _draw_glow_text(img, (MX + w // 2, bar_y + bar_h // 2), _format_number(val),
                            fval, (0, 0, 0), color, glow_radius=0)
        else:
            draw.text((MX + w + 24, bar_y + bar_h // 2), _format_number(val),
                      font=fval, fill=color, anchor='lm')
        fpct, _ = _fit_font(draw, f'{pct:.0f}%', 45, 120, bold=True)
        draw.text((MX + CW, bar_y + bar_h // 2), f'{pct:.0f}%', font=fpct,
                  fill=color, anchor='rm')

    winner = u1_clean if m1 >= m2 else u2_clean
    diff = abs(m1 - m2) / max(m1, m2) * 100 if max(m1, m2) else 0
    insight = f'{winner} пишет активнее на {diff:.0f}%'
    fins, flines, _ = _fit_wrapped(draw, insight, 48, CW, max_lines=2)
    centered_block(img, draw, 1520, flines, fins, text_muted)

    buf = io.BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
    slides.append(buf)

    # ============ СЛАЙД 3: ТОП-5 СЛОВ ОБОИХ ============
    img, draw = new_slide()
    _draw_orb(img, 100, 200, 200, accent1, alpha=40)
    _draw_orb(img, 980, 1700, 220, accent2, alpha=40)
    slide_title(draw, 'ГЛАВНЫЕ СЛОВА')

    draw.line([(W // 2, 330), (W // 2, H - 180)], fill=divider, width=2)
    col_w = CW // 2 - 50
    for cx, name, words, color in [
        (MX + CW // 4, u1_clean.upper(), tw1_list, accent1),
        (MX + 3 * CW // 4, u2_clean.upper(), tw2_list, accent2),
    ]:
        fname, _ = _fit_font(draw, name, 48, col_w, bold=True)
        draw.text((cx, 380), name, font=fname, fill=color, anchor='mm')
        y = 520
        for word, count in words:
            fword, _ = _fit_font(draw, word.upper(), 64, col_w, bold=True)
            _draw_glow_text(img, (cx, y), word.upper(), fword, text_main, color,
                            glow_radius=max(6, fword.size // 9))
            cnt = f'{_format_number(count)} {_pluralize(count, "раз", "раза", "раз")}'
            fcnt, _ = _fit_font(draw, cnt, 40, col_w)
            draw.text((cx, y + fword.size // 2 + 40), cnt, font=fcnt, fill=text_muted, anchor='mm')
            y += 190
        if not words:
            fempty, _ = _fit_font(draw, 'нет слов', 40, col_w)
            draw.text((cx, 600), 'нет слов', font=fempty, fill=text_muted, anchor='mm')

    buf = io.BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
    slides.append(buf)

    # ============ СЛАЙД 4: ЗАТРАЧЕННОЕ ВРЕМЯ ============
    img, draw = new_slide()
    _draw_orb(img, 900, 400, 250, accent1, alpha=50)
    _draw_orb(img, 150, 1500, 200, accent2, alpha=50)
    slide_title(draw, 'ВРЕМЯ В ЧАТЕ')

    total_hours = time_info['total_session_hours']
    hours_int = int(total_hours)
    hstr = _format_number(hours_int)
    fh, _ = _fit_font(draw, hstr, 150, CW, bold=True)
    _draw_glow_text(img, (W // 2, 500), hstr, fh, text_main, accent1, glow_radius=15)
    sub = f'{_pluralize(hours_int, "час", "часа", "часов")} активного общения'
    fsub, _ = _fit_font(draw, sub, 45, CW)
    draw.text((W // 2, 640), sub, font=fsub, fill=text_muted, anchor='mm')

    span_str = f'За период: {_format_timedelta(time_info["span"])}'
    fspan, slines, _ = _fit_wrapped(draw, span_str, 42, CW, max_lines=2)
    centered_block(img, draw, 800, slines, fspan, text_muted)

    draw.text((W // 2, 980), 'ЗА ЭТО ВРЕМЯ МОЖНО БЫЛО:', font=fsub, fill=accent2, anchor='mm')

    comparisons = _creative_comparisons(total_hours)
    y = 1120
    for comp in comparisons:
        fitem, clines, _ = _fit_wrapped(draw, comp, 42, CW, max_lines=2)
        centered_block(img, draw, y + (len(clines) - 1) * 27, clines, fitem, text_main)
        y += 150 if len(clines) > 1 else 110

    buf = io.BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
    slides.append(buf)

    # ============ СЛАЙД 5: ПИК АКТИВНОСТИ ============
    img, draw = new_slide()
    _draw_orb(img, 900, 500, 220, accent1, alpha=50)
    _draw_orb(img, 150, 1400, 180, accent2, alpha=50)
    slide_title(draw, 'ВЫ ЧАЩЕ ВСЕГО ПИШЕТЕ В')

    time_str = f'{peak_hour:02d}:00'
    ft, _ = _fit_font(draw, time_str, 260, CW, bold=True)
    _draw_glow_text(img, (W // 2, 620), time_str, ft, text_main, accent1, glow_radius=20)

    msg_line = f'{_format_number(peak_count)} {_pluralize(peak_count, "сообщение", "сообщения", "сообщений")}'
    fmc, _ = _fit_font(draw, msg_line, 50, CW, bold=True)
    draw.text((W // 2, 830), msg_line, font=fmc, fill=text_main, anchor='mm')
    fmm, _ = _fit_font(draw, 'в этот час', 42, CW)
    draw.text((W // 2, 905), 'в этот час', font=fmm, fill=text_muted, anchor='mm')

    hourly_sorted = df.group_by('hour').agg(pl.len().alias('count')).sort('hour').to_pandas()
    hourly_sorted = hourly_sorted.set_index('hour').reindex(range(24)).fillna(0)
    counts = hourly_sorted['count'].tolist()
    max_c = max(counts) if counts else 1
    graph_w = CW
    graph_h = 280
    graph_x = MX
    graph_y = 1050

    points = []
    for i, c in enumerate(counts):
        x = graph_x + (i / 23) * graph_w
        yv = graph_y + graph_h - (c / max_c) * graph_h
        points.append((x, yv))
    if len(points) > 1:
        draw.line(points, fill=accent1, width=6)
        peak_idx = counts.index(max_c)
        px_p, py_p = points[peak_idx]
        draw.ellipse((px_p - 15, py_p - 15, px_p + 15, py_p + 15), fill=accent1)

    fsmall, _ = _fit_font(draw, '00', 32, 60)
    for h in [0, 6, 12, 18, 23]:
        x = graph_x + (h / 23) * graph_w
        draw.text((x, graph_y + graph_h + 30), f'{h:02d}', font=fsmall, fill=text_muted, anchor='mm')

    buf = io.BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
    slides.append(buf)

    # ============ СЛАЙД 6: АКТИВНЫЙ ДЕНЬ ============
    img, draw = new_slide()
    _draw_orb(img, 540, 400, 300, accent2, alpha=40)
    slide_title(draw, 'САМЫЙ АКТИВНЫЙ ДЕНЬ')

    fday, _ = _fit_font(draw, peak_day, 140, CW, bold=True)
    _draw_glow_text(img, (W // 2, 580), peak_day, fday, text_main, accent2, glow_radius=18)

    avg_line = f'~{_format_number(peak_day_avg)} {_pluralize(peak_day_avg, "сообщение", "сообщения", "сообщений")}'
    favg, _ = _fit_font(draw, avg_line, 55, CW, bold=True)
    draw.text((W // 2, 810), avg_line, font=favg, fill=text_main, anchor='mm')
    fcap, _ = _fit_font(draw, 'в этот день в среднем', 42, CW)
    draw.text((W // 2, 890), 'в этот день в среднем', font=fcap, fill=text_muted, anchor='mm')

    pct_str = f'{peak_day_pct:.1f}%'
    fpct, _ = _fit_font(draw, pct_str, 100, CW, bold=True)
    _draw_glow_text(img, (W // 2, 1060), pct_str, fpct, accent2, accent2, glow_radius=10)
    draw.text((W // 2, 1160), 'всех сообщений чата', font=fcap, fill=text_muted, anchor='mm')

    if second_day_ru and diff_pct > 0:
        cmp_line = f'на {diff_pct:.0f}% больше, чем {second_day_ru}'
        fcmp, _ = _fit_font(draw, cmp_line, 40, CW)
        draw.text((W // 2, 1260), cmp_line, font=fcmp, fill=text_muted, anchor='mm')

    days_short = ['ПН', 'ВТ', 'СР', 'ЧТ', 'ПТ', 'СБ', 'ВС']
    days_full = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    circle_r = 40
    spacing = (CW - 2 * circle_r) // 6
    circle_y = 1520
    start_x = W // 2 - 3 * spacing
    fshort, _ = _fit_font(draw, 'ПН', 28, 2 * circle_r, bold=True)
    for i, (short, full) in enumerate(zip(days_short, days_full)):
        cx = start_x + i * spacing
        is_peak = (full == weekday['weekday_name'][0]) if weekday.height > 0 else False
        color = accent2 if is_peak else divider
        draw.ellipse((cx - circle_r, circle_y - circle_r, cx + circle_r, circle_y + circle_r), fill=color)
        draw.text((cx, circle_y), short, font=fshort,
                  fill=(0, 0, 0) if is_peak else text_muted, anchor='mm')

    buf = io.BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
    slides.append(buf)

    # ============ СЛАЙД 7: САМЫЙ ДЛИННЫЙ ДИАЛОГ ============
    img, draw = new_slide()
    _draw_orb(img, 200, 400, 220, accent1, alpha=50)
    _draw_orb(img, 880, 1500, 250, accent2, alpha=50)
    slide_title(draw, 'САМЫЙ ДЛИННЫЙ ДИАЛОГ')

    duration_str = _format_timedelta(longest['duration'])
    fdur, dlines, _ = _fit_wrapped(draw, duration_str, 110, CW, bold=True, max_lines=2)
    centered_block(img, draw, 560, dlines, fdur, text_main, glow=accent1, gap=1.2)

    msg_count = longest['messages']
    mc_line = f'{msg_count} {_pluralize(msg_count, "сообщение", "сообщения", "сообщений")}'
    fmc, _ = _fit_font(draw, mc_line, 50, CW, bold=True)
    draw.text((W // 2, 800), mc_line, font=fmc, fill=text_main, anchor='mm')

    fdt, _ = _fit_font(draw, 'Начало: 00.00.0000 00:00', 42, CW)
    if longest['start'] and longest['end']:
        start_str = longest['start'].strftime('%d.%m.%Y %H:%M')
        end_str = longest['end'].strftime('%d.%m.%Y %H:%M')
        s1, _ = _fit_font(draw, f'Начало: {start_str}', 42, CW)
        draw.text((W // 2, 940), f'Начало: {start_str}', font=s1, fill=text_muted, anchor='mm')
        s2, _ = _fit_font(draw, f'Конец: {end_str}', 42, CW)
        draw.text((W // 2, 1010), f'Конец: {end_str}', font=s2, fill=text_muted, anchor='mm')

    draw.line([(W // 2 - 250, 1130), (W // 2 + 250, 1130)], fill=accent2, width=2)

    ffact, _ = _fit_font(draw, 'Непрерывный обмен', 38, CW)
    draw.text((W // 2, 1230), 'Непрерывный обмен', font=ffact, fill=text_muted, anchor='mm')
    draw.text((W // 2, 1300), 'без пауз более 30 минут', font=ffact, fill=text_muted, anchor='mm')

    buf = io.BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
    slides.append(buf)

    # ============ СЛАЙД 8: НОЧНАЯ АКТИВНОСТЬ ============
    img, draw = new_slide()
    _draw_orb(img, 540, 300, 280, (100, 100, 200), alpha=40)
    _draw_orb(img, 200, 1500, 180, accent1, alpha=40)
    slide_title(draw, 'НОЧНАЯ АКТИВНОСТЬ')
    fmm2, _ = _fit_font(draw, '23:00 — 05:00', 42, CW)
    draw.text((W // 2, 300), '23:00 — 05:00', font=fmm2, fill=text_muted, anchor='mm')

    pct_n = f'{night["night_pct"]:.1f}%'
    fpn, _ = _fit_font(draw, pct_n, 180, CW, bold=True)
    _draw_glow_text(img, (W // 2, 540), pct_n, fpn, text_main, accent1, glow_radius=20)
    fcap2, _ = _fit_font(draw, 'сообщений ночью', 42, CW)
    draw.text((W // 2, 690), 'сообщений ночью', font=fcap2, fill=text_muted, anchor='mm')

    nc, tc = night['night_count'], night['total']
    col_w8 = CW // 2 - 40
    fnum8, _ = _fit_font(draw, _format_number(nc), 70, col_w8, bold=True)
    draw.text((MX + col_w8 // 2, 900), _format_number(nc), font=fnum8, fill=accent1, anchor='mm')
    draw.text((MX + col_w8 // 2, 985), 'ночных', font=fcap2, fill=text_muted, anchor='mm')
    fnum8b, _ = _fit_font(draw, _format_number(tc), 70, col_w8, bold=True)
    draw.text((MX + col_w8 + col_w8 // 2, 900), _format_number(tc), font=fnum8b, fill=accent2, anchor='mm')
    draw.text((MX + col_w8 + col_w8 // 2, 985), 'всего', font=fcap2, fill=text_muted, anchor='mm')

    ph_str = f'{night["peak_hour"]:02d}:00'
    fph, _ = _fit_font(draw, ph_str, 100, CW, bold=True)
    _draw_glow_text(img, (W // 2, 1230), ph_str, fph, text_main, accent2, glow_radius=15)
    fcap3, _ = _fit_font(draw, 'самый активный ночной час', 42, CW)
    draw.text((W // 2, 1330), 'самый активный ночной час', font=fcap3, fill=text_muted, anchor='mm')
    npc_line = f'{_format_number(night["peak_count"])} {_pluralize(night["peak_count"], "сообщение", "сообщения", "сообщений")}'
    fnpc, _ = _fit_font(draw, npc_line, 50, CW, bold=True)
    draw.text((W // 2, 1420), npc_line, font=fnpc, fill=text_main, anchor='mm')

    buf = io.BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
    slides.append(buf)

    # ============ СЛАЙД 9: ФИНАЛЬНЫЙ ============
    img, draw = new_slide()
    _draw_orb(img, 200, 400, 200, accent1, alpha=60)
    _draw_orb(img, 850, 1500, 250, accent2, alpha=60)
    _draw_orb(img, 900, 300, 150, (255, 255, 255), alpha=40)
    _draw_orb(img, 150, 1600, 180, (255, 255, 255), alpha=30)

    final_names = f'{u1_clean} × {u2_clean}'
    ffn, fnlines, _ = _fit_wrapped(draw, final_names, 130, CW, bold=True, max_lines=2)
    centered_block(img, draw, 640, fnlines, ffn, text_main, glow=accent1, gap=1.2)

    draw.line([(W // 2 - 250, 830), (W // 2 + 250, 830)], fill=accent2, width=3)

    metrics = [
        f'{_format_number(total_msgs)} {_pluralize(total_msgs, "сообщение", "сообщения", "сообщений")}',
        f'{_format_number(total_reactions)} {_pluralize(total_reactions, "реакция", "реакции", "реакций")}',
        f'{_format_number(total_gifts)} {_pluralize(total_gifts, "подарок", "подарка", "подарков")}',
        f'{_format_number(total_replies)} {_pluralize(total_replies, "ответ", "ответа", "ответов")}',
    ]
    y = 980
    for line in metrics:
        fm, _ = _fit_font(draw, line, 46, CW, bold=True)
        draw.text((W // 2, y), line, font=fm, fill=text_main, anchor='mm')
        y += 105

    gen_line = f'Сгенерировано {datetime.now().strftime("%d.%m.%Y")}'
    fg, _ = _fit_font(draw, gen_line, 50, CW)
    draw.text((W // 2, 1680), gen_line, font=fg, fill=text_muted, anchor='mm')

    buf = io.BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
    slides.append(buf)

    return slides
# ---------- ЗАГРУЗКА ФАЙЛА ----------
# Лого и название в сайдбаре — на одной строке, лого выровнено по тексту
if os.path.exists("hydrox_logo.jpg"):
    with open("hydrox_logo.jpg", "rb") as _lf:
        logo_b64 = base64.b64encode(_lf.read()).decode()
    st.sidebar.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:14px; padding:4px 0 16px 0;">
            <img src="data:image/jpeg;base64,{logo_b64}"
                 style="width:52px; height:52px; border-radius:14px; object-fit:cover; margin:0; flex:0 0 auto;">
            <div style="line-height:1.15;">
                <div style="font-size:1.5rem; font-weight:700;">Hydrox</div>
                <div style="color:gray; font-size:0.85rem;">анализ вашего чата</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        "<h2 style='margin-bottom: 0;'>Hydrox</h2>"
        "<p style='color: gray; margin-top: 0;'>анализ вашего чата</p>",
        unsafe_allow_html=True,
    )

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
                df_ps = pd.DataFrame(part_stats).T.reset_index().rename(columns={
                    'index': 'Участник',
                    'messages': 'Сообщений',
                    'share_pct': 'Доля, %',
                    'text_messages': 'Текстовых',
                    'replies': 'Ответов',
                    'media_messages': 'Медиа',
                    'edited': 'Изменено',
                    'avg_words': 'Ср. слов',
                    'avg_chars': 'Ср. символов'
                })
                df_ps['Доля, %'] = df_ps['Доля, %'].round(1)
                df_ps['Ср. слов'] = df_ps['Ср. слов'].round(1)
                df_ps['Ср. символов'] = df_ps['Ср. символов'].round(1)
                st.dataframe(df_ps, use_container_width=True, hide_index=True)
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
            hourly = df_filtered.group_by(['hour', 'sender']).agg(pl.len().alias('count')).sort('hour').to_pandas()
            fig = px.line(hourly, x='hour', y='count', color='sender',
                          title='Сообщения по часам', markers=True,
                          color_discrete_sequence=[colors['user1'], colors['user2']])
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Тепловая карта: день недели / час")
            pivot = df_filtered.group_by(['weekday_name', 'hour']).agg(pl.len().alias('count')).to_pandas()
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
                gap_df = pd.DataFrame(gap_results).T.reset_index().rename(columns={
                    'index': 'Автор',
                    'turns': 'Смен автора',
                    'median': 'Медиана, мин',
                    'mean': 'Среднее, мин',
                    'p75': 'P75, мин',
                    'p90': 'P90, мин'
                }).round(2)
                st.dataframe(gap_df, use_container_width=True, hide_index=True)
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
                daily_display = daily_pd.rename(columns={
                    'day': 'Дата', 'total': 'Всего сообщений', 'users': 'Активных участников',
                    'reactions': 'Реакций', 'photos': 'Фото', 'videos': 'Видео',
                    'audios': 'Аудио', 'texts': 'Текстов', 'stickers': 'Стикеров',
                    'files': 'Файлов', 'words': 'Слов', 'replies': 'Ответов',
                    'gifts': 'Подарков', 'service_count': 'Сервисных записей',
                    'peak_hour': 'Пиковый час'
                })
                st.dataframe(daily_display, use_container_width=True, hide_index=True)
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
                monthly_display = monthly_pd.rename(columns={
                    'year_month': 'Месяц', 'messages': 'Сообщений', 'words': 'Слов',
                    'replies': 'Ответов', 'media': 'Медиа', 'reactions': 'Реакций',
                    'edits': 'Изменено', 'days_active': 'Дней активности', 'gifts': 'Подарков'
                })
                st.dataframe(monthly_display, use_container_width=True, hide_index=True)
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
                hourly = df_filtered.group_by(['hour', 'sender']).agg(pl.len().alias('count')).sort('hour').to_pandas()
                fig = px.line(hourly, x='hour', y='count', color='sender',
                              title='Активность по часам',
                              color_discrete_sequence=[colors['user1'], colors['user2']])
                st.plotly_chart(fig, use_container_width=True)
                weekday = df_filtered.group_by(['weekday_name', 'sender']).agg(pl.len().alias('count')).sort('weekday_name').to_pandas()
                fig = px.bar(weekday, x='weekday_name', y='count', color='sender', barmode='group',
                             title='Активность по дням недели',
                             color_discrete_sequence=[colors['user1'], colors['user2']])
                st.plotly_chart(fig, use_container_width=True)
                media_by_user = df_filtered.group_by(['sender', 'media_type']).agg(pl.len().alias('count')).sort('count').to_pandas()
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
                if slope > 0.5:
                    trend_label = "растёт 📈"
                elif slope < -0.5:
                    trend_label = "снижается 📉"
                else:
                    trend_label = "стабильна ➡️"
                st.write(f"Тренд: {slope:+.2f} сообщений в день — активность {trend_label}")
                st.write(f"Базовый уровень: {intercept:.0f} сообщений в день")
                st.caption("Прогноз строится по 7-дневному сглаживанию: тренд ограничен, "
                           "поэтому случайный шум не «уронит» прогноз в ноль.")
                daily_hist_pd = daily_hist.to_pandas()
                pred_pd = pred_df.to_pandas()
                fig = px.line(daily_hist_pd, x='day', y='count')
                fig.update_traces(opacity=0.3)
                fig.add_scatter(x=daily_hist_pd['day'], y=daily_hist_pd['smoothed'],
                                mode='lines', name='Сглаженный ряд (7 дней)',
                                line=dict(color=colors['user2']))
                fig.add_scatter(x=pred_pd['date'], y=pred_pd['predicted'], mode='lines+markers',
                                name='Прогноз', line=dict(dash='dash', color='red'))
                fig.update_layout(title='Ежедневная активность и прогноз',
                                  xaxis_title='Дата', yaxis_title='Сообщений')
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(pred_pd)
            else:
                st.info("Недостаточно данных для прогноза.")

else:
    st.info("Загрузите JSON-файл экспорта чата Telegram для начала анализа.")
    with st.expander("📖 Как экспортировать чат из Telegram", expanded=True):
        st.markdown(
            """
**Только Telegram Desktop** (мобильная версия не умеет экспортировать):

1. Откройте **Telegram Desktop** на компьютере.
2. Зайдите в **Настройки → Экспортировать историю**.
3. Выберите нужный чат и снимите со всех пунктов галочки.
4. Формат обязательно — **JSON** (не HTML!).
5. Нажмите **«Экспорт»** и дождитесь окончания.
6. Найдите файл `result.json` в папке с экспортом и загрузите его в панель слева.

> ⚠️ Файл обрабатывается на сервере и после обработки удаляется.
"""
        )

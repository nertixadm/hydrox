import json
import polars as pl
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import re
import yaml
import numpy as np
import os
from sklearn.linear_model import LinearRegression

# ---------- ЗАГРУЗКА КОНФИГА ----------
def load_config(config_path="config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# ---------- ЗАГРУЗКА ДАННЫХ ----------
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        if 'messages' in data:
            messages = data['messages']
        elif 'result' in data and isinstance(data['result'], dict) and 'messages' in data['result']:
            messages = data['result']['messages']
        elif 'chat' in data and 'messages' in data['chat']:
            messages = data['chat']['messages']
        else:
            found = False
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict) and 'id' in value[0]:
                    messages = value
                    found = True
                    break
            if not found:
                raise ValueError("Неизвестный формат JSON: не найдено поле с сообщениями")
    elif isinstance(data, list):
        messages = data
    else:
        raise ValueError("Неизвестный формат JSON")
    return messages

# ---------- ПАРСИНГ ----------
def parse_messages(messages):
    records = []
    append = records.append
    for msg in messages:
        if msg.get('type') == 'service':
            continue
        dt = datetime.fromisoformat(msg['date'].replace('Z', '+00:00'))
        sender = msg.get('from', 'Unknown')
        text = msg.get('text', '')
        if isinstance(text, list):
            text = ''.join([part if isinstance(part, str) else part.get('text', '') 
                            for part in text])
        media_type = msg.get('media_type', None)
        if media_type is None:
            if 'photo' in msg:
                media_type = 'photo'
            elif 'video' in msg:
                media_type = 'video'
            elif 'audio' in msg:
                media_type = 'audio'
            elif 'sticker' in msg:
                media_type = 'sticker'
            elif 'gif' in msg:
                media_type = 'gif'
            elif 'voice' in msg:
                media_type = 'voice'
            elif 'video_note' in msg:
                media_type = 'video_note'
            elif 'contact' in msg:
                media_type = 'contact'
            elif 'location' in msg:
                media_type = 'location'
            elif 'poll' in msg:
                media_type = 'poll'
            elif 'file' in msg:
                media_type = 'file'
            elif 'gift' in msg:
                media_type = 'gift'
            else:
                media_type = 'text'
        reactions = msg.get('reactions', [])
        reactions_count = sum(r.get('count', 0) for r in reactions) if isinstance(reactions, list) else 0
        reply_to = msg.get('reply_to_message_id', None)
        edited = 1 if 'edited' in msg else 0
        msg_id = msg.get('id', None)
        append({
            'id': msg_id,
            'date': dt,
            'sender': sender,
            'text': text,
            'media_type': media_type,
            'reactions_count': reactions_count,
            'reactions_list': reactions if isinstance(reactions, list) else [],
            'reply_to': reply_to,
            'edited': edited,
            'words_count': len(text.split()) if text else 0
        })
    df = pl.DataFrame(records)
    if df.height > 0:
        df = df.with_columns([
            pl.col('date').cast(pl.Datetime),
            pl.col('date').dt.date().alias('day'),
            pl.col('date').dt.hour().alias('hour'),
            pl.col('date').dt.weekday().alias('weekday'),
            pl.col('date').dt.strftime('%A').alias('weekday_name'),
            pl.col('date').dt.strftime('%Y-%m').alias('year_month')
        ])
    return df

# ---------- ПОДАРКИ ПО ДНЯМ ----------
def gifts_by_day(raw_messages):
    gifts = defaultdict(int)
    for msg in raw_messages:
        if msg.get('type') == 'service' and msg.get('action') == 'send_star_gift':
            dt = datetime.fromisoformat(msg['date'].replace('Z', '+00:00'))
            gifts[dt.date()] += 1
    if gifts:
        return pl.Series(name='day', values=list(gifts.keys()), dtype=pl.Date), \
               pl.Series(name='gifts', values=list(gifts.values()), dtype=pl.Int32)
    else:
        return pl.Series(name='day', dtype=pl.Date), pl.Series(name='gifts', dtype=pl.Int32)

# ---------- СЕРВИСНЫЕ ЗАПИСИ ПО ДНЯМ ----------
def service_by_day(raw_messages):
    service = defaultdict(int)
    for msg in raw_messages:
        if msg.get('type') == 'service':
            dt = datetime.fromisoformat(msg['date'].replace('Z', '+00:00'))
            service[dt.date()] += 1
    if service:
        return pl.Series(name='day', values=list(service.keys()), dtype=pl.Date), \
               pl.Series(name='service_count', values=list(service.values()), dtype=pl.Int32)
    else:
        return pl.Series(name='day', dtype=pl.Date), \
               pl.Series(name='service_count', dtype=pl.Int32)

# ---------- ОБЩАЯ СТАТИСТИКА ----------
def general_stats(df, gifts_total):
    total = df.height
    sender_counts = df.group_by('sender').agg(pl.len().alias('count')).to_dict(as_series=False)
    sender_counts = dict(zip(sender_counts['sender'], sender_counts['count']))
    media_counts = df.group_by('media_type').agg(pl.len().alias('count')).to_dict(as_series=False)
    media_counts = dict(zip(media_counts['media_type'], media_counts['count']))
    text_msgs = df.filter(pl.col('media_type') == 'text')
    avg_len = text_msgs['text'].str.len_chars().mean() if text_msgs.height > 0 else 0
    total_reactions = df['reactions_count'].sum()
    replies = df['reply_to'].is_not_null().sum()
    edited = df['edited'].sum()
    return {
        'total_messages': total,
        'sender_counts': sender_counts,
        'media_counts': media_counts,
        'avg_text_length': avg_len,
        'total_reactions': total_reactions,
        'gifts': gifts_total,
        'replies': replies,
        'edited': edited
    }

# ---------- КОНТРОЛЬ КАЧЕСТВА ----------
def data_quality_check(df, raw_messages):
    total_raw = len(raw_messages)
    ids = [m.get('id') for m in raw_messages if 'id' in m]
    unique_ids = len(set(ids))
    duplicates = len(ids) - unique_ids
    dates_parsed = df.height
    is_ordered = df['date'].is_sorted() if df.height > 0 else True
    return {
        'total_records': total_raw,
        'unique_ids': unique_ids,
        'duplicates': duplicates,
        'dates_parsed': dates_parsed,
        'ordered': is_ordered
    }

# ---------- СТАТИСТИКА ПО УЧАСТНИКАМ ----------
def participant_stats(df):
    total_messages = df.height
    stats = {}
    for sender in df['sender'].unique().to_list():
        sub = df.filter(pl.col('sender') == sender)
        total = sub.height
        texts = sub.filter(pl.col('media_type') == 'text')
        text_count = texts.height
        replies = sub['reply_to'].is_not_null().sum()
        media = sub.filter(pl.col('media_type') != 'text').height
        edited = sub['edited'].sum()
        avg_words = texts['words_count'].mean() if text_count > 0 else 0
        avg_chars = texts['text'].str.len_chars().mean() if text_count > 0 else 0
        stats[sender] = {
            'messages': total,
            'share_pct': total / total_messages * 100 if total_messages > 0 else 0,
            'text_messages': text_count,
            'replies': replies,
            'media_messages': media,
            'edited': edited,
            'avg_words': avg_words,
            'avg_chars': avg_chars
        }
    return stats

# ---------- РЕАКЦИИ ПО ТИПАМ ----------
def reactions_breakdown(df):
    all_reactions = Counter()
    for row in df.iter_rows(named=True):
        for r in row['reactions_list']:
            emoji = r.get('emoji', r.get('type', 'unknown'))
            count = r.get('count', 1)
            all_reactions[emoji] += count
    return dict(all_reactions)

# ---------- ДЛИТЕЛЬНОСТЬ ЗВОНКОВ ----------
def call_duration_stats(raw_messages):
    durations = []
    for msg in raw_messages:
        if msg.get('type') != 'service':
            continue
        action = msg.get('action', '')
        if action in ('phone_call', 'call'):
            dur = None
            if 'duration_seconds' in msg:
                dur = msg['duration_seconds']
            elif 'duration' in msg:
                dur = msg['duration']
            elif 'call' in msg and isinstance(msg['call'], dict):
                dur = msg['call'].get('duration')
            if dur is not None:
                try:
                    durations.append(int(dur))
                except (ValueError, TypeError):
                    pass
    if durations:
        return {
            'total_calls': len(durations),
            'total_duration_sec': sum(durations),
            'avg_duration_sec': sum(durations) / len(durations),
            'durations_list': durations,
            'max_duration': max(durations),
            'min_duration': min(durations)
        }
    return None

# ---------- МЕСЯЧНАЯ СТАТИСТИКА ----------
def monthly_analytics(df, gift_df):
    if df.height == 0:
        return pl.DataFrame()
    monthly = df.group_by('year_month').agg([
        pl.len().alias('messages'),
        pl.sum('words_count').alias('words'),
        pl.col('reply_to').is_not_null().sum().alias('replies'),
        (pl.col('media_type') != 'text').cast(pl.Int32).sum().alias('media'),
        pl.sum('reactions_count').alias('reactions'),
        pl.sum('edited').alias('edits'),
        pl.n_unique('day').alias('days_active')
    ]).sort('year_month')
    if gift_df is not None and gift_df.height > 0:
        gifts_grouped = gift_df.with_columns(
            pl.col('day').cast(pl.Date).dt.strftime('%Y-%m').alias('year_month')
        ).group_by('year_month').agg(pl.sum('gifts').alias('gifts'))
        monthly = monthly.join(gifts_grouped, on='year_month', how='left').fill_null(0)
    else:
        monthly = monthly.with_columns(pl.lit(0).alias('gifts'))
    return monthly

# ---------- АНАЛИЗ ИНТЕРВАЛОВ ПО АВТОРАМ ----------
def reply_gap_by_sender(df, max_gap_hours=24):
    df_sorted = df.sort('date')
    gaps = []
    prev_row = None
    for row in df_sorted.iter_rows(named=True):
        if prev_row is not None and row['sender'] != prev_row['sender']:
            gap = (row['date'] - prev_row['date']).total_seconds() / 60
            if gap <= max_gap_hours * 60:
                gaps.append((prev_row['sender'], gap))
        prev_row = row
    results = {}
    for sender in df['sender'].unique().to_list():
        sender_gaps = [g for s, g in gaps if s == sender]
        if sender_gaps:
            results[sender] = {
                'turns': len(sender_gaps),
                'median': np.median(sender_gaps),
                'mean': np.mean(sender_gaps),
                'p75': np.percentile(sender_gaps, 75),
                'p90': np.percentile(sender_gaps, 90)
            }
    return results

# ---------- СЕССИИ ----------
def analyze_sessions(df, gap_hours=8):
    df_sorted = df.sort('date')
    sessions = []
    for sender in df['sender'].unique().to_list():
        group = df_sorted.filter(pl.col('sender') == sender).sort('date')
        session_start = None
        session_end = None
        prev_time = None
        for row in group.iter_rows(named=True):
            current_time = row['date']
            if prev_time is None:
                session_start = current_time
                session_end = current_time
            else:
                gap = (current_time - prev_time).total_seconds() / 3600
                if gap > gap_hours:
                    duration = (session_end - session_start).total_seconds() / 3600
                    session_msgs = group.filter(
                        (pl.col('date') >= session_start) & (pl.col('date') <= session_end)
                    ).height
                    sessions.append({
                        'sender': sender,
                        'start': session_start,
                        'end': session_end,
                        'duration_hours': duration,
                        'messages': session_msgs
                    })
                    session_start = current_time
                    session_end = current_time
                else:
                    session_end = current_time
            prev_time = current_time
        if session_start is not None:
            duration = (session_end - session_start).total_seconds() / 3600
            session_msgs = group.filter(
                (pl.col('date') >= session_start) & (pl.col('date') <= session_end)
            ).height
            sessions.append({
                'sender': sender,
                'start': session_start,
                'end': session_end,
                'duration_hours': duration,
                'messages': session_msgs
            })
    return pl.DataFrame(sessions)

# ---------- ЕЖЕДНЕВНАЯ АНАЛИТИКА (с опцией include_service) ----------
def daily_analytics(df, gift_df, service_df=None, include_service=False):
    if df.height == 0:
        return pl.DataFrame()
    daily = df.group_by('day').agg([
        pl.len().alias('total'),
        pl.n_unique('sender').alias('users'),
        pl.sum('reactions_count').alias('reactions'),
        (pl.col('media_type') == 'photo').cast(pl.Int32).sum().alias('photos'),
        (pl.col('media_type') == 'video').cast(pl.Int32).sum().alias('videos'),
        (pl.col('media_type') == 'audio').cast(pl.Int32).sum().alias('audios'),
        (pl.col('media_type') == 'text').cast(pl.Int32).sum().alias('texts'),
        (pl.col('media_type') == 'sticker').cast(pl.Int32).sum().alias('stickers'),
        (pl.col('media_type') == 'file').cast(pl.Int32).sum().alias('files'),
        pl.sum('words_count').alias('words'),
        pl.col('reply_to').is_not_null().sum().alias('replies')
    ]).sort('day')

    daily = daily.with_columns(
        pl.col('day').cast(pl.Date).dt.strftime('%A').alias('weekday')
    )

    senders = df['sender'].unique().to_list()
    for sender in senders:
        sender_df = df.filter(pl.col('sender') == sender).group_by('day').agg(pl.len().alias('count'))
        daily = daily.join(sender_df, on='day', how='left').rename({'count': f'from_{sender}'}).fill_null(0)

    peak_hours = df.group_by(['day', 'hour']).agg(pl.len().alias('count')).sort(['day', 'count'], descending=[False, True])
    peak_hours = peak_hours.unique(subset=['day'], keep='first').select(['day', 'hour']).rename({'hour': 'peak_hour'})
    daily = daily.join(peak_hours, on='day', how='left').fill_null(0)

    if gift_df is not None and gift_df.height > 0:
        daily = daily.join(gift_df, on='day', how='left').fill_null(0)
    else:
        daily = daily.with_columns(pl.lit(0).alias('gifts'))

    # Опционально добавляем сервисные записи к общей сумме
    if include_service and service_df is not None and service_df.height > 0:
        daily = daily.join(service_df, on='day', how='left').fill_null(0)
        daily = daily.with_columns(
            (pl.col('total') + pl.col('service_count')).alias('total')
        )
    else:
        daily = daily.with_columns(pl.lit(0).alias('service_count'))

    return daily

# ---------- ПРОГНОЗИРОВАНИЕ (двойное экспоненциальное сглаживание Холта) ----------
def _holt_fit(y, alpha, beta):
    """Один проход алгоритма Холта; возвращает SSE, финальный уровень и тренд."""
    level = y[0]
    trend = y[1] - y[0] if len(y) > 1 else 0.0
    sse = 0.0
    for t in range(1, len(y)):
        forecast = level + trend
        sse += (y[t] - forecast) ** 2
        prev_level = level
        level = alpha * y[t] + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    return sse, level, trend


def predict_activity(df, days_ahead=7):
    if df.height == 0:
        return None, None, None, None
    daily = df.group_by('day').agg(pl.len().alias('count')).sort('day')
    min_day = daily['day'].min()
    max_day = daily['day'].max()
    span_days = (max_day - min_day).days
    if span_days < 2:
        return None, None, None, None

    # Полный календарный ряд: дни без сообщений = 0 (иначе Holt сжимает паузы)
    all_days = [min_day + timedelta(days=i) for i in range(span_days + 1)]
    full = (
        pl.DataFrame({'day': all_days})
        .join(daily, on='day', how='left')
        .fill_null(0)
        .with_columns([
            pl.col('count').cast(pl.Float64),
            (pl.col('day') - min_day).dt.total_days().cast(pl.Int32).alias('day_num'),
        ])
    )

    # Сглаживание шума: скользящее среднее за 7 дней
    full = full.with_columns(
        pl.col('count').rolling_mean(window_size=7, min_samples=1).alias('smoothed')
    )
    y = full['smoothed'].to_numpy().astype(float)

    if len(y) >= 3:
        # Подбираем alpha и beta по минимуму SSE на обучающей выборке
        best = None
        for alpha in np.arange(0.1, 1.0, 0.1):
            for beta in np.arange(0.05, 1.0, 0.05):
                sse, level, trend = _holt_fit(y, float(alpha), float(beta))
                if best is None or sse < best[0]:
                    best = (sse, level, trend)
        _, level, trend = best
    else:
        # Недостаточно данных — скользящее среднее
        level = float(np.mean(y))
        trend = 0.0

    # Защита от вырожденного тренда: наклон ограничен ±5% среднего уровня в день,
    # чтобы шум не «уронил» прогноз в ноль
    mean_level = float(np.mean(y)) if len(y) else 0.0
    max_trend = mean_level * 0.05
    trend = float(np.clip(trend, -max_trend, max_trend))

    slope = float(trend)
    intercept = float(level)
    last_day_num = int(full['day_num'].max())

    # Затухание тренда (damping): прогноз сходится к уровню, а не улетает в ноль
    phi = 0.85
    horizons = np.arange(1, days_ahead + 1)
    damp_factors = phi * (1 - phi ** horizons) / (1 - phi)
    pred_counts = np.maximum(level + trend * damp_factors, 0).round().astype(int)

    # Прогноз не должен уходить ниже разумной доли недавнего среднего уровня
    recent_mean = float(np.mean(y[-7:])) if len(y) >= 7 else mean_level
    pred_counts = np.maximum(pred_counts, int(recent_mean * 0.3))

    pred_dates = [min_day + timedelta(days=last_day_num + d) for d in range(1, days_ahead + 1)]
    pred_df = pl.DataFrame({'date': pred_dates, 'predicted': pred_counts})
    return pred_df, slope, intercept, full

# ---------- ЧАСТОТНЫЕ СЛОВА ----------# ---------- ЧАСТОТНЫЕ СЛОВА ----------
def analyze_frequent_words(df, stop_words, top_n=10, min_len=3):
    text_df = df.filter(pl.col('media_type') == 'text')
    if text_df.height == 0:
        return None
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = [w for w in text.split() if w not in stop_words and len(w) >= min_len]
        return words
    user_word_counts = {}
    all_counter = Counter()
    for sender in df['sender'].unique().to_list():
        sender_texts = text_df.filter(pl.col('sender') == sender)['text'].to_list()
        sender_counter = Counter()
        for t in sender_texts:
            words = clean_text(t)
            sender_counter.update(words)
            all_counter.update(words)
        user_word_counts[sender] = sender_counter.most_common(top_n)
    overall_top = all_counter.most_common(top_n)
    return {
        'overall': overall_top,
        'users': user_word_counts,
        'all_counter': all_counter
    }

# ---------- БИГРАММЫ ----------
def analyze_bigrams(df, stop_words, min_len=3, top_n=20):
    text_df = df.filter(pl.col('media_type') == 'text')
    if text_df.height == 0:
        return []
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = [w for w in text.split() if w not in stop_words and len(w) >= min_len]
        return words
    bigram_counter = Counter()
    for t in text_df['text'].to_list():
        words = clean_text(t)
        for i in range(len(words)-1):
            bigram = (words[i], words[i+1])
            bigram_counter[bigram] += 1
    return bigram_counter.most_common(top_n)

# ---------- СЕРВИСНЫЕ ЗАПИСИ ----------
def service_actions_analysis(raw_messages):
    service_msgs = [m for m in raw_messages if m.get('type') != 'message']
    action_counts = Counter()
    for m in service_msgs:
        action = m.get('action', None)
        if action:
            action_counts[action] += 1
        else:
            action_counts['unknown'] += 1
    return {
        'total_service': len(service_msgs),
        'action_counts': dict(action_counts)
    }

# ---------- РАСПРЕДЕЛЕНИЕ ДЛИНЫ ТЕКСТА ----------
def text_length_distribution(df):
    text_df = df.filter(pl.col('media_type') == 'text')
    if text_df.height == 0:
        return pl.Series(), pl.Series()
    word_counts = text_df['words_count']
    char_counts = text_df['text'].str.len_chars()
    return word_counts, char_counts

# -*- coding: utf-8 -*-
import os, logging, smtplib, traceback, re
from datetime import datetime
from dateutil import parser
from email.mime.text import MIMEText
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

# 服务器端生成图片所需非交互式后端
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# --- 配置 -------------------------------------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SMTP_SERVER    = "smtp.gmail.com"
SMTP_PORT      = 587
SMTP_USERNAME = "kata.chatbot@gmail.com"
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

# --- 語言常數 ---------------------------------------------------------
LANGUAGE = {
    "tw": {
        "email_subject": "您的健康洞察報告",
        "report_title" : "🎉 全球健康洞察報告"
    }
}

LANGUAGE_TEXTS = {
    "tw": {
        "name": "法定全名", "chinese_name": "中文姓名", "dob": "出生日期", "country": "國家",
        "gender": "性別", "age": "年齡", "height": "身高 (公分)", "weight": "體重 (公斤)",
        "concern": "主要問題", "details": "補充說明", "referrer": "推薦人", "angel": "健康夥伴",
        "footer": "📩 此報告已透過電子郵件傳送給您。所有內容均由 KataChat AI 生成，並符合個人資料保護法規定。"
    }
}

# --- 工具函數 ---------------------------------------------------------
def compute_age(dob):
    try:
        dt = parser.parse(dob)
        today = datetime.today()
        return today.year - dt.year - ((today.month, today.day) < (dt.month, dt.day))
    except:
        return 0

# --- AI 提示 (已修訂) --------------------------------------------------
def build_summary_prompt(age, gender, country, concern, notes, metrics):
    """
    生成四段式健康分析提示：
    - 使用群體化描述，避免第二人稱與個人化措辭
    - 每段需自然嵌入百分比
    """
    metrics_summary = ", ".join([
        f"{label} ({value}%)"
        for block in metrics
        for label, value in zip(block["labels"], block["values"])
    ][:9])

    return (
        f"任務：針對來自 {country}、年齡約 {age} 歲的 {gender} 群體，撰寫一份四段式健康分析，"
        f"其主要問題為「{concern}」。請使用以下數據：{metrics_summary}。\n\n"
        f"指令：\n"
        f"1. **深入分析**：不要只重複數據，闡述這些比例對該群體意味著什麼，並分析指標之間的關聯。\n"
        f"2. **內容豐富**：每段都提供有價值的背景資訊，語氣同理心且專業。\n"
        f"3. **匿名措辭**：嚴禁出現第二人稱或「該用戶/個體」，用「類似年齡段的 {country} {gender}」等表述。\n"
        f"4. **整合數據**：每段自然融入至少一個具體百分比。\n"
    )

def build_suggestions_prompt(age, gender, country, concern, notes):
    """
    生成 10 條生活方式建議（群體化措辭，禁止客套開場）
    """
    return (
        f"為來自 {country}、年齡約 {age} 歲、關注「{concern}」的 {gender} 群體，"
        f"提出 10 項具體而溫和的生活方式改善建議。\n"
        f"請使用溫暖、支持的語氣，並加入合適的表情符號。\n"
        f"⚠️ **嚴格指令**：\n"
        f"- 不得使用姓名、第二人稱，也不要出現「當然可以」等客套開頭。\n"
        f"- 用「在 {country}，同年齡段的 {gender} 群體…」等群體化表述。\n"
    )

# --- 與 OpenAI 互動 ----------------------------------------------------
def get_openai_response(prompt, temp=0.7):
    try:
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp
        )
        return result.choices[0].message.content
    except Exception as e:
        logging.error(f"OpenAI error: {e}")
        return "⚠️ 無法生成回應。"

def generate_metrics_with_ai(prompt):
    """
    生成並解析 3 組指標區塊，格式:
    ### 指標標題
    指標A: 65%
    指標B: 78%
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        lines = res.choices[0].message.content.strip().split("\n")
        metrics, current_title, labels, values = [], "", [], []
        for line in lines:
            if line.startswith("###"):
                if current_title:
                    metrics.append({"title": current_title, "labels": labels, "values": values})
                current_title, labels, values = line.replace("###", "").strip(), [], []
            elif ":" in line:
                try:
                    label, val = line.split(":", 1)
                    labels.append(label.strip())
                    values.append(int(val.strip().replace("%", "")))
                except ValueError:
                    continue
        if current_title:
            metrics.append({"title": current_title, "labels": labels, "values": values})
        return metrics or [{"title": "預設指標", "labels": ["指標A", "指標B"], "values": [50, 75]}]
    except Exception as e:
        logging.error(f"Chart parse error: {e}")
        return [{"title": "預設指標", "labels": ["指標A", "指標B"], "values": [50, 75]}]

# --- HTML 及郵件生成 ----------------------------------------------------
def generate_user_data_html(user_info, labels):
    html = """
    <h2 style="font-family: sans-serif; color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 5px;">個人資料摘要</h2>
    <table style="width: 100%; border-collapse: collapse; font-family: sans-serif; margin-bottom: 30px;">
    """
    display_order = [
        'name', 'chinese_name', 'age', 'gender', 'country',
        'height', 'weight', 'condition', 'details', 'referrer', 'angel'
    ]
    for key in display_order:
        value = user_info.get(key)
        label_text = labels.get(key, key.replace('_', ' ').title())
        if value:
            html += f"""
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 12px; background-color: #f9f9f9; font-weight: bold; width: 150px;">{label_text}</td>
                <td style="padding: 12px;">{value}</td>
            </tr>
            """
    html += "</table>"
    return html

def generate_custom_charts_html(metrics):
    charts_html = '<h2 style="font-family: sans-serif; color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 5px;">健康指標圖表</h2>'
    for metric in metrics:
        charts_html += f'<h3 style="font-family: sans-serif; color: #333; margin-top: 20px;">{metric["title"]}</h3>'
        for label, value in zip(metric["labels"], metric["values"]):
            charts_html += f"""
            <div style="margin-bottom: 12px; font-family: sans-serif;">
                <p style="margin: 0 0 5px 0;">- {label}: {value}%</p>
                <div style="background-color: #e0e0e0; border-radius: 8px; width: 100%; height: 16px;">
                    <div style="background-color: #4CAF50; width: {value}%; height: 16px; border-radius: 8px;"></div>
                </div>
            </div>
            """
    return charts_html

def generate_footer_html():
    return """
    <div style="margin-top: 40px; border-left: 4px solid #4CAF50; padding-left: 15px; font-family: sans-serif;">
        <h3 style="font-size: 22px; font-weight: bold; color: #333;">📊 由 KataChat AI 生成的洞察</h3>
        <p style="font-size: 18px; color: #555; line-height: 1.6;">
            此健康報告是使用 KataChat 的專有 AI 模型生成的，基於：
        </p>
        <ul style="list-style-type: disc; padding-left: 20px; font-size: 18px; color: #555; line-height: 1.6;">
            <li>來自新加坡、馬來西亞和台灣用戶的匿名健康與生活方式資料庫</li>
            <li>來自可信的 OpenAI 研究資料庫的全球健康基準和行為趨勢數據</li>
        </ul>
        <p style="font-size: 18px; color: #555; line-height: 1.6;">
            所有分析嚴格遵守個人資料保護法規，以保護您的個人資料，同時發掘有意義的健康洞察。
        </p>
        <p style="font-size: 18px; color: #555; line-height: 1.6; margin-top: 15px;">
            🛡️ <strong>請注意：</strong>本報告並非醫療診斷。若有任何嚴重的健康問題，請諮詢持牌醫療專業人員。
        </p>
        <p style="font-size: 18px; color: #555; line-height: 1.6; margin-top: 15px;">
            📬 <strong>附註：</strong>個人化報告將在 24-48 小時內傳送到您的電子信箱。若您想更詳細地探討報告結果，我們很樂意安排一個 15 分鐘的簡短通話。
        </p>
    </div>
    """

def send_email_report(recipient_email, subject, body):
    if not all([SMTP_SERVER, SMTP_USERNAME, SMTP_PASSWORD]):
        logging.warning("SMTP settings are not fully configured. Skipping email.")
        return
    try:
        msg = MIMEText(body, 'html', 'utf-8')
        msg['Subject'] = subject
        msg['From'] = f"KataChat AI <{SMTP_USERNAME}>"
        msg['To'] = recipient_email

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_USERNAME, [recipient_email], msg.as_string())
            logging.info(f"Successfully sent health report to {recipient_email}")
    except Exception as e:
        logging.error(f"Failed to send email to {recipient_email}: {e}")
        traceback.print_exc()

# --- Flask 路由 --------------------------------------------------------
@app.route("/health_analyze", methods=["POST"])
def health_analyze():
    try:
        data = request.get_json(force=True)
        lang = data.get("lang", "tw").strip().lower()
        if lang != "tw":
            return jsonify({"error": "此端點僅支援台灣繁體中文 (tw)。"}), 400

        labels       = LANGUAGE_TEXTS[lang]
        content_lang = LANGUAGE[lang]

        dob = f"{data.get('dob_year')}-{str(data.get('dob_month')).zfill(2)}-{str(data.get('dob_day')).zfill(2)}"
        age = compute_age(dob)

        user_info = {k: data.get(k) for k in [
            "name", "chinese_name", "gender", "height", "weight",
            "country", "condition", "referrer", "angel", "details"
        ]}
        user_info.update({"dob": dob, "age": age, "notes": data.get("details") or "無補充說明"})

        # --- AI 生成 ----------------------------------------------------
        chart_prompt = (
            f"這是一位來自 {user_info['country']} 的 {user_info['age']} 歲 {user_info['gender']}，"
            f"其健康問題為「{user_info['condition']}'。補充說明：{user_info['notes']}\n\n"
            f"請根據此問題生成 3 個不同的健康相關指標類別。\n"
            f"每個類別必須以 '###' 開頭（例如 '### 睡眠品質'），並包含 3 個獨特的真實世界指標，格式為 '指標名稱: 68%'.\n"
            f"所有百分比必須介於 25% 到 90% 之間。\n"
            f"僅返回 3 個格式化的區塊，不要有任何介紹或解釋。"
        )
        metrics = generate_metrics_with_ai(chart_prompt)

        summary_prompt     = build_summary_prompt(age, user_info['gender'], user_info['country'],
                                                  user_info['condition'], user_info['notes'], metrics)
        summary            = get_openai_response(summary_prompt)

        suggestions_prompt = build_suggestions_prompt(age, user_info['gender'], user_info['country'],
                                                        user_info['condition'], user_info['notes'])
        creative           = get_openai_response(suggestions_prompt, temp=0.85)
        # --- 客套開場剝離（如仍出現「當然可以！」） ---------------------
        creative = re.sub(r"^\s*當然可以[！!]\s*", "", creative)

        # --- 建構郵件 HTML ------------------------------------------
        email_html_body = f"""
        <div style='font-family: sans-serif; color: #333; max-width: 800px; margin: auto; padding: 20px;'>
            <h1 style='text-align:center; color: #333;'>{content_lang['report_title']}</h1>

            {generate_user_data_html(user_info, labels)}

            {generate_custom_charts_html(metrics)}

            <div style="margin-top: 30px;">
                <h2 style="font-family: sans-serif; color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 5px;">🧠 摘要</h2>
                {''.join([f"<p style='line-height:1.7; font-size:16px;'>{p.strip()}</p>" for p in summary.strip().split('  ') if p.strip()])}
            </div>

            <div style="margin-top: 30px;">
                <h2 style="font-family: sans-serif; color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 5px;">💡 生活建議</h2>
                {''.join([f"<p style='margin:12px 0; font-size:16px; line-height:1.6;'>{line}</p>" for line in creative.splitlines() if line.strip()])}
            </div>

            {generate_footer_html()}
        </div>
        """

        # --- 發送郵件 ----------------------------------------------
        email_subject = f"{content_lang['email_subject']} - {user_info.get('name', 'N/A')}"
        send_email_report(SMTP_USERNAME, email_subject, email_html_body)

        # --- 返回前端所需結構（Chart.js 仍適用） ------------------
        html_result_for_web  = "<div style='font-family: sans-serif; color: #333;'>"
        html_result_for_web += "<div style='font-size:24px; font-weight:bold; margin-top:30px;'>🧠 摘要:</div>"
        html_result_for_web += "".join([
            f"<p style='line-height:1.7; font-size:16px; margin-top:1em; margin-bottom:1em;'>{p.strip()}</p>"
            for p in summary.strip().split('\n\n') if p.strip()
        ])
        html_result_for_web += "<div style='font-size:24px; font-weight:bold; margin-top:40px;'>💡 生活建議:</div>"
        html_result_for_web += "".join([
            f"<p style='margin:16px 0; font-size:17px; line-height:1.6;'>{line}</p>"
            for line in creative.split("\n") if line.strip()
        ])
        html_result_for_web += generate_footer_html() + "</div>"

        return jsonify({
            "metrics"     : metrics,
            "html_result" : html_result_for_web,
            "footer"      : labels['footer'],
            "report_title": content_lang['report_title']
        })

    except Exception as e:
        logging.error(f"Health analyze error: {e}")
        traceback.print_exc()
        return jsonify({"error": "發生未預期的伺服器錯誤。"}), 500

# ----------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False, port=int(os.getenv("PORT", 5000)), host="0.0.0.0")

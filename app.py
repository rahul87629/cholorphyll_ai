"""
app.py — LeafSense v6
✓ Multilingual: English, Hindi, Tamil, Telugu, Kannada
✓ No keyboard_double_arrow text — hidden with CSS
✓ Creative farmer content, rich imagery, professional UI
✓ Light / Dark mode
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os

st.set_page_config(
    page_title="LeafSense · Precision Agriculture",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# TRANSLATIONS
# ══════════════════════════════════════════════════════════════════════════════
LANG = {
    "English": {
        "app_name": "LeafSense",
        "tagline": "PRECISION AGRICULTURE AI",
        "nav_predict": "🔬 Predict",
        "nav_dataset": "📊 Dataset",
        "nav_guide":   "📖 SPAD Guide",
        "nav_about":   "⚙️ About",
        "nav_label":   "NAVIGATION",
        "settings":    "SETTINGS",
        "dark_mode":   "🌙 Dark Mode",
        "auto_save":   "💾 Auto-save",
        "show_feats":  "🎨 Colour Features",
        "language":    "🌐 Language",
        "model_ok":    "✓ MODEL READY",
        "model_err":   "✗ MODEL NOT FOUND\nRun: python train.py",
        "hero_title":  "🌾 Chlorophyll SPAD Predictor",
        "hero_sub":    "Upload a rice leaf · AI analyses chlorophyll · Get instant agronomic advice",
        "upload_head": "Upload Leaf Image",
        "upload_hint": "Supports JPG · PNG · BMP",
        "tips_head":   "📌 Capture Tips",
        "tips":        ["White or plain background", "Natural daylight, no shadows", "20–30 cm from leaf", "Full leaf with veins visible"],
        "await_title": "Awaiting Leaf Image",
        "await_sub":   "Upload on the left to begin analysis",
        "analysing":   "Analysing leaf …",
        "spad_label":  "PREDICTED SPAD VALUE",
        "spad_unit":   "CHLOROPHYLL INDEX · SPAD UNITS",
        "health_lbl":  "LEAF HEALTH",
        "n_lbl":       "EST. LEAF NITROGEN",
        "advice_head": "AGRONOMIC RECOMMENDATION",
        "saved_msg":   "✓ Saved to dataset",
        "status": {
            "CRITICAL":  "🔴 Severe nitrogen deficiency. Apply 30 kg N/ha urea immediately. Re-test in 5 days.",
            "POOR":      "🟠 Nitrogen stress. Apply split dose 20 kg N/ha. Check soil pH (optimal 6.0–7.0).",
            "MODERATE":  "🟡 Sub-optimal. Light topdress 10–15 kg N/ha. Monitor every 7 days.",
            "GOOD":      "🟢 Healthy chlorophyll. Continue current fertilisation schedule.",
            "EXCELLENT": "💚 High chlorophyll. Vigorous crop. No nitrogen action needed.",
        },
        "guide_title": "📖 Farmer's SPAD Reference Guide",
        "guide_sub":   "Chlorophyll interpretation · Nitrogen management · Best practices",
        "table_headers": ["SPAD Range","Status","Leaf Colour","Cause","Action"],
        "table_rows": [
            ["0–15","🔴 Critical","Severely yellow","Severe N deficiency","30 kg N/ha immediately"],
            ["15–25","🟠 Poor","Pale yellow-green","N shortage","Split 20 kg N/ha + check pH"],
            ["25–35","🟡 Moderate","Light green","Sub-optimal","Light topdress, monitor weekly"],
            ["35–45","🟢 Good","Normal green","Healthy","Continue schedule"],
            ["45–80","💚 Excellent","Dark green","High chlorophyll","No action needed"],
        ],
        "tips_timing": ["Measure 7–10 AM for stable readings","Use topmost fully expanded leaf","3 readings per plant — average them","Same leaf position every time"],
        "tips_stage":  ["Tillering: 35–40 SPAD","Panicle initiation: 38–42 SPAD","Heading: 35–38 SPAD"],
        "tips_n":      ["Apply urea (46-0-0) at 20–25 kg/ha","Split into 2 doses, 1 week apart","Re-measure after 7 days","Irrigate after application"],
        "tips_warn":   ["SPAD > 50 → lodging risk","Excess N attracts brown plant hopper","Promotes blast disease","Always verify with soil test"],
        "farmer_quotes": [
            '"A healthy leaf means a healthy harvest." — Traditional farming wisdom',
            '"Soil is the foundation of all farming. Chlorophyll is the mirror of soil health."',
            '"Measure first. Fertilise wisely. Save cost and environment."',
        ],
        "did_you_know": [
            "Rice provides 20% of the world's dietary energy supply.",
            "A 1-unit drop in SPAD can reduce rice yield by up to 1.5%.",
            "Proper nitrogen management can reduce fertilizer waste by 30–40%.",
            "India is the world's second-largest producer of rice after China.",
        ],
    },
    "हिन्दी": {
        "app_name": "लीफसेंस",
        "tagline": "सटीक कृषि AI",
        "nav_predict": "🔬 पूर्वानुमान",
        "nav_dataset": "📊 डेटासेट",
        "nav_guide":   "📖 SPAD गाइड",
        "nav_about":   "⚙️ परिचय",
        "nav_label":   "नेविगेशन",
        "settings":    "सेटिंग्स",
        "dark_mode":   "🌙 डार्क मोड",
        "auto_save":   "💾 स्वतः सहेजें",
        "show_feats":  "🎨 रंग विशेषताएं",
        "language":    "🌐 भाषा",
        "model_ok":    "✓ मॉडल तैयार",
        "model_err":   "✗ मॉडल नहीं मिला\npython train.py चलाएं",
        "hero_title":  "🌾 क्लोरोफिल SPAD पूर्वानुमान",
        "hero_sub":    "पत्ती की फोटो अपलोड करें · AI क्लोरोफिल का विश्लेषण करे · तुरंत सलाह पाएं",
        "upload_head": "पत्ती की छवि अपलोड करें",
        "upload_hint": "JPG · PNG · BMP समर्थित",
        "tips_head":   "📌 फोटो टिप्स",
        "tips":        ["सफेद या सादा पृष्ठभूमि", "प्राकृतिक रोशनी, छाया नहीं", "पत्ती से 20–30 सेमी दूरी", "पूरी पत्ती दिखनी चाहिए"],
        "await_title": "पत्ती की छवि की प्रतीक्षा",
        "await_sub":   "विश्लेषण शुरू करने के लिए बाईं ओर अपलोड करें",
        "analysing":   "पत्ती का विश्लेषण हो रहा है …",
        "spad_label":  "अनुमानित SPAD मान",
        "spad_unit":   "क्लोरोफिल सूचकांक · SPAD इकाइयां",
        "health_lbl":  "पत्ती स्वास्थ्य",
        "n_lbl":       "अनुमानित पत्ती नाइट्रोजन",
        "advice_head": "कृषि सुझाव",
        "saved_msg":   "✓ डेटासेट में सहेजा गया",
        "status": {
            "CRITICAL":  "🔴 गंभीर नाइट्रोजन की कमी। तुरंत 30 किग्रा N/हेक्टेयर यूरिया डालें।",
            "POOR":      "🟠 नाइट्रोजन तनाव। 20 किग्रा N/हेक्टेयर विभाजित खुराक में डालें।",
            "MODERATE":  "🟡 उप-इष्टतम। 10–15 किग्रा N/हेक्टेयर हल्की खुराक दें।",
            "GOOD":      "🟢 स्वस्थ क्लोरोफिल। वर्तमान उर्वरक कार्यक्रम जारी रखें।",
            "EXCELLENT": "💚 उच्च क्लोरोफिल। स्वस्थ फसल। नाइट्रोजन की जरूरत नहीं।",
        },
        "guide_title": "📖 किसान SPAD संदर्भ मार्गदर्शिका",
        "guide_sub":   "क्लोरोफिल व्याख्या · नाइट्रोजन प्रबंधन · सर्वोत्तम प्रथाएं",
        "table_headers": ["SPAD सीमा","स्थिति","पत्ती रंग","कारण","कार्रवाई"],
        "table_rows": [
            ["0–15","🔴 गंभीर","गहरी पीली","N की भारी कमी","तुरंत 30 किग्रा N/हे"],
            ["15–25","🟠 खराब","हल्की पीली","N की कमी","20 किग्रा N/हे + pH जांचें"],
            ["25–35","🟡 मध्यम","हल्की हरी","उप-इष्टतम","हल्की खुराक, साप्ताहिक निगरानी"],
            ["35–45","🟢 अच्छा","सामान्य हरा","स्वस्थ","कार्यक्रम जारी रखें"],
            ["45–80","💚 उत्कृष्ट","गहरी हरी","उच्च क्लोरोफिल","कोई कार्रवाई नहीं"],
        ],
        "tips_timing": ["सुबह 7–10 बजे माप लें","सबसे ऊपरी पूर्ण पत्ती का उपयोग करें","3 रीडिंग लें और औसत निकालें","हर बार एक ही स्थान पर मापें"],
        "tips_stage":  ["कल्लेदार अवस्था: 35–40 SPAD","बाली निकलने से पहले: 38–42 SPAD","सिर निकलने पर: 35–38 SPAD"],
        "tips_n":      ["यूरिया (46-0-0) 20–25 किग्रा/हेक्टेयर","2 खुराकों में, 1 सप्ताह अंतराल","7 दिन बाद पुनः माप लें","डालने के बाद सिंचाई करें"],
        "tips_warn":   ["SPAD > 50 → गिरने का खतरा","अधिक N → भूरा पौधा होपर आकर्षित","ब्लास्ट रोग को बढ़ावा","मिट्टी परीक्षण से सत्यापित करें"],
        "farmer_quotes": [
            '"स्वस्थ पत्ती का अर्थ है भरपूर फसल।" — कृषि परंपरा',
            '"मिट्टी खेती की नींव है, क्लोरोफिल उसका दर्पण।"',
            '"पहले माप लो, फिर बुद्धिमानी से उर्वरक डालो।"',
        ],
        "did_you_know": [
            "चावल विश्व की 20% आहार ऊर्जा प्रदान करता है।",
            "SPAD में 1 इकाई की गिरावट से उपज 1.5% तक कम हो सकती है।",
            "उचित नाइट्रोजन प्रबंधन से उर्वरक की 30–40% बचत होती है।",
            "भारत चीन के बाद विश्व का दूसरा सबसे बड़ा चावल उत्पादक है।",
        ],
    },
    "தமிழ்": {
        "app_name": "லீஃப்சென்ஸ்",
        "tagline": "துல்லிய விவசாய AI",
        "nav_predict": "🔬 கணிப்பு",
        "nav_dataset": "📊 தரவுத்தொகுப்பு",
        "nav_guide":   "📖 SPAD வழிகாட்டி",
        "nav_about":   "⚙️ பற்றி",
        "nav_label":   "வழிசெலுத்தல்",
        "settings":    "அமைப்புகள்",
        "dark_mode":   "🌙 இருண்ட பயன்முறை",
        "auto_save":   "💾 தானாக சேமி",
        "show_feats":  "🎨 வண்ண அம்சங்கள்",
        "language":    "🌐 மொழி",
        "model_ok":    "✓ மாதிரி தயார்",
        "model_err":   "✗ மாதிரி இல்லை\npython train.py இயக்கவும்",
        "hero_title":  "🌾 குளோரோபில் SPAD கணிப்பு",
        "hero_sub":    "இலை படம் பதிவேற்றவும் · AI பகுப்பாய்வு · உடனடி விவசாய ஆலோசனை",
        "upload_head": "இலை படம் பதிவேற்றவும்",
        "upload_hint": "JPG · PNG · BMP ஆதரிக்கப்படுகிறது",
        "tips_head":   "📌 படம் எடுக்கும் குறிப்புகள்",
        "tips":        ["வெள்ளை அல்லது எளிய பின்னணி", "இயற்கை வெளிச்சம், நிழல் வேண்டாம்", "இலையிலிருந்து 20–30 செமீ தூரம்", "முழு இலையும் தெரிய வேண்டும்"],
        "await_title": "இலை படத்திற்காக காத்திருக்கிறோம்",
        "await_sub":   "பகுப்பாய்வு தொடங்க இடதுபுறம் பதிவேற்றவும்",
        "analysing":   "இலை பகுப்பாய்வு நடக்கிறது …",
        "spad_label":  "கணிக்கப்பட்ட SPAD மதிப்பு",
        "spad_unit":   "குளோரோபில் குறியீடு · SPAD அலகுகள்",
        "health_lbl":  "இலை ஆரோக்கியம்",
        "n_lbl":       "மதிப்பிடப்பட்ட நைட்ரஜன்",
        "advice_head": "விவசாய பரிந்துரை",
        "saved_msg":   "✓ தரவுத்தொகுப்பில் சேமிக்கப்பட்டது",
        "status": {
            "CRITICAL":  "🔴 கடுமையான நைட்ரஜன் பற்றாக்குறை. உடனடியாக 30 கிகி N/ஹெக்டேர் யூரியா இடவும்.",
            "POOR":      "🟠 நைட்ரஜன் அழுத்தம். 20 கிகி N/ஹெ பிரித்து இடவும்.",
            "MODERATE":  "🟡 சரியான அளவில் இல்லை. 10–15 கிகி N/ஹெ லேசாக இடவும்.",
            "GOOD":      "🟢 ஆரோக்கியமான குளோரோபில். தற்போதைய திட்டத்தை தொடரவும்.",
            "EXCELLENT": "💚 அதிக குளோரோபில். வலுவான பயிர். நைட்ரஜன் தேவையில்லை.",
        },
        "guide_title": "📖 விவசாயி SPAD குறிப்பு வழிகாட்டி",
        "guide_sub":   "குளோரோபில் விளக்கம் · நைட்ரஜன் மேலாண்மை · சிறந்த நடைமுறைகள்",
        "table_headers": ["SPAD வரம்பு","நிலை","இலை வண்ணம்","காரணம்","நடவடிக்கை"],
        "table_rows": [
            ["0–15","🔴 மிகவும் மோசம்","மஞ்சள்","கடுமையான N பற்றாக்குறை","30 கிகி N/ஹெ உடனடியாக"],
            ["15–25","🟠 மோசம்","வெளிர் மஞ்சள்-பச்சை","N குறைபாடு","20 கிகி N/ஹெ + pH சரிபார்க்கவும்"],
            ["25–35","🟡 சராசரி","வெளிர் பச்சை","சரியான அளவில் இல்லை","லேசாக இடவும், வாராந்திரம் கண்காணிக்கவும்"],
            ["35–45","🟢 நல்லது","சாதாரண பச்சை","ஆரோக்கியமான","திட்டத்தை தொடரவும்"],
            ["45–80","💚 மிகவும் நல்லது","அடர் பச்சை","அதிக குளோரோபில்","நடவடிக்கை தேவையில்லை"],
        ],
        "tips_timing": ["காலை 7–10 மணிக்கு அளவிடவும்","மேலே உள்ள முழு இலையை பயன்படுத்தவும்","3 அளவீடுகள் எடுத்து சராசரி கணக்கிடவும்","ஒவ்வொரு முறையும் ஒரே இடத்தில் அளவிடவும்"],
        "tips_stage":  ["கன்னல் நிலை: 35–40 SPAD","கதிர் தொடக்கம்: 38–42 SPAD","கதிர் நிலை: 35–38 SPAD"],
        "tips_n":      ["யூரியா 20–25 கிகி/ஹெ இடவும்","2 தவணைகளில், 1 வார இடைவெளியில்","7 நாட்களுக்கு பின் மீண்டும் அளவிடவும்","இட்ட பிறகு நீர்ப்பாசனம் செய்யவும்"],
        "tips_warn":   ["SPAD > 50 → பயிர் விழும் ஆபத்து","அதிக N → பழுப்பு பூச்சி ஈர்க்கப்படும்","பிளாஸ்ட் நோய் வருவதற்கு வழிவகுக்கும்","மண் பரிசோதனை மூலம் உறுதிப்படுத்தவும்"],
        "farmer_quotes": [
            '"ஆரோக்கியமான இலை என்றால் நல்ல அறுவடை." — விவசாய ஞானம்',
            '"மண் விவசாயத்தின் அடித்தளம், குளோரோபில் அதன் கண்ணாடி."',
            '"முதலில் அளவிடு, பின்னர் புத்திசாலித்தனமாக உரமிடு."',
        ],
        "did_you_know": [
            "அரிசி உலகின் 20% உணவு ஆற்றலை வழங்குகிறது.",
            "SPAD 1 அலகு குறைந்தால் மகசூல் 1.5% குறையலாம்.",
            "சரியான நைட்ரஜன் மேலாண்மை உரச் செலவை 30–40% குறைக்கும்.",
            "இந்தியா சீனாவுக்கு அடுத்தபடியாக இரண்டாவது பெரிய அரிசி உற்பத்தியாளர்.",
        ],
    },
    "తెలుగు": {
        "app_name": "లీఫ్‌సెన్స్",
        "tagline": "ఖచ్చితమైన వ్యవసాయ AI",
        "nav_predict": "🔬 అంచనా",
        "nav_dataset": "📊 డేటాసెట్",
        "nav_guide":   "📖 SPAD గైడ్",
        "nav_about":   "⚙️ గురించి",
        "nav_label":   "నావిగేషన్",
        "settings":    "సెట్టింగ్‌లు",
        "dark_mode":   "🌙 డార్క్ మోడ్",
        "auto_save":   "💾 స్వయంచాలక సేవ్",
        "show_feats":  "🎨 రంగు లక్షణాలు",
        "language":    "🌐 భాష",
        "model_ok":    "✓ మోడల్ సిద్ధంగా ఉంది",
        "model_err":   "✗ మోడల్ లేదు\npython train.py నడపండి",
        "hero_title":  "🌾 క్లోరోఫిల్ SPAD అంచనా",
        "hero_sub":    "ఆకు ఫోటో అప్‌లోడ్ చేయండి · AI క్లోరోఫిల్ విశ్లేషిస్తుంది · తక్షణ సలహా పొందండి",
        "upload_head": "ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి",
        "upload_hint": "JPG · PNG · BMP మద్దతు ఉంది",
        "tips_head":   "📌 ఫోటో చిట్కాలు",
        "tips":        ["తెల్లని లేదా సాదా నేపథ్యం", "సహజ కాంతి, నీడలు వద్దు", "ఆకు నుండి 20–30 సెమీ దూరం", "మొత్తం ఆకు కనిపించాలి"],
        "await_title": "ఆకు చిత్రం కోసం వేచి ఉంది",
        "await_sub":   "విశ్లేషణ ప్రారంభించడానికి ఎడమవైపు అప్‌లోడ్ చేయండి",
        "analysing":   "ఆకు విశ్లేషిస్తోంది …",
        "spad_label":  "అంచనా SPAD విలువ",
        "spad_unit":   "క్లోరోఫిల్ సూచిక · SPAD యూనిట్లు",
        "health_lbl":  "ఆకు ఆరోగ్యం",
        "n_lbl":       "అంచనా నత్రజని",
        "advice_head": "వ్యవసాయ సూచన",
        "saved_msg":   "✓ డేటాసెట్‌లో సేవ్ చేయబడింది",
        "status": {
            "CRITICAL":  "🔴 తీవ్రమైన నత్రజని లోపం. వెంటనే 30 కి.గ్రా N/హె యూరియా వేయండి.",
            "POOR":      "🟠 నత్రజని ఒత్తిడి. 20 కి.గ్రా N/హె విభజించి వేయండి.",
            "MODERATE":  "🟡 సరిపోని స్థాయి. 10–15 కి.గ్రా N/హె తేలికగా వేయండి.",
            "GOOD":      "🟢 ఆరోగ్యకరమైన క్లోరోఫిల్. ప్రస్తుత షెడ్యూల్ కొనసాగించండి.",
            "EXCELLENT": "💚 అధిక క్లోరోఫిల్. శక్తివంతమైన పంట. నత్రజని అవసరం లేదు.",
        },
        "guide_title": "📖 రైతు SPAD సూచన మార్గదర్శి",
        "guide_sub":   "క్లోరోఫిల్ వివరణ · నత్రజని నిర్వహణ · ఉత్తమ పద్ధతులు",
        "table_headers": ["SPAD పరిధి","స్థితి","ఆకు రంగు","కారణం","చర్య"],
        "table_rows": [
            ["0–15","🔴 తీవ్రం","ముదురు పసుపు","తీవ్ర N లోపం","వెంటనే 30 కి.గ్రా N/హె"],
            ["15–25","🟠 పేద","లేత పసుపు-ఆకుపచ్చ","N లోపం","20 కి.గ్రా N/హె + pH తనిఖీ"],
            ["25–35","🟡 మధ్యస్థ","లేత ఆకుపచ్చ","సరిపోని","తేలికగా వేయండి, వారంవారీ పర్యవేక్షించండి"],
            ["35–45","🟢 మంచి","సాధారణ ఆకుపచ్చ","ఆరోగ్యకరం","షెడ్యూల్ కొనసాగించండి"],
            ["45–80","💚 అద్భుతం","ముదురు ఆకుపచ్చ","అధిక క్లోరోఫిల్","చర్య అవసరం లేదు"],
        ],
        "tips_timing": ["ఉదయం 7–10 గంటలకు కొలవండి","పై పూర్తి ఆకును ఉపయోగించండి","3 రీడింగ్‌లు తీసుకుని సగటు వేయండి","ప్రతిసారీ అదే స్థానంలో కొలవండి"],
        "tips_stage":  ["పిలకల దశ: 35–40 SPAD","పానికిల్ దశ: 38–42 SPAD","హెడింగ్ దశ: 35–38 SPAD"],
        "tips_n":      ["యూరియా 20–25 కి.గ్రా/హె వేయండి","2 డోసులు, 1 వారం వ్యవధిలో","7 రోజుల తర్వాత మళ్లీ కొలవండి","వేసిన తర్వాత నీరు పెట్టండి"],
        "tips_warn":   ["SPAD > 50 → లాడ్జింగ్ ప్రమాదం","అధిక N → బ్రౌన్ ప్లాంట్‌హాపర్","బ్లాస్ట్ వ్యాధి వచ్చే అవకాశం","మట్టి పరీక్ష ద్వారా నిర్ధారించండి"],
        "farmer_quotes": [
            '"ఆరోగ్యకరమైన ఆకు అంటే సమృద్ధిగా పంట." — సంప్రదాయ జ్ఞానం',
            '"మట్టి వ్యవసాయానికి పునాది, క్లోరోఫిల్ దాని అద్దం."',
            '"ముందు కొలవండి, తెలివిగా ఎరువు వేయండి."',
        ],
        "did_you_know": [
            "వరి ప్రపంచ ఆహార శక్తిలో 20% అందిస్తుంది.",
            "SPAD 1 యూనిట్ తగ్గితే దిగుబడి 1.5% తక్కువ అవుతుంది.",
            "సరైన నత్రజని నిర్వహణ 30–40% ఎరువు ఖర్చు తగ్గిస్తుంది.",
            "భారతదేశం చైనా తర్వాత రెండవ అతిపెద్ద వరి ఉత్పత్తిదారు.",
        ],
    },
    "ಕನ್ನಡ": {
        "app_name": "ಲೀಫ್‌ಸೆನ್ಸ್",
        "tagline": "ನಿಖರ ಕೃಷಿ AI",
        "nav_predict": "🔬 ಮುನ್ಸೂಚನೆ",
        "nav_dataset": "📊 ಡೇಟಾಸೆಟ್",
        "nav_guide":   "📖 SPAD ಮಾರ್ಗದರ್ಶಿ",
        "nav_about":   "⚙️ ಬಗ್ಗೆ",
        "nav_label":   "ನ್ಯಾವಿಗೇಷನ್",
        "settings":    "ಸೆಟ್ಟಿಂಗ್‌ಗಳು",
        "dark_mode":   "🌙 ಡಾರ್ಕ್ ಮೋಡ್",
        "auto_save":   "💾 ಸ್ವಯಂ ಉಳಿಸಿ",
        "show_feats":  "🎨 ಬಣ್ಣ ಲಕ್ಷಣಗಳು",
        "language":    "🌐 ಭಾಷೆ",
        "model_ok":    "✓ ಮಾದರಿ ಸಿದ್ಧ",
        "model_err":   "✗ ಮಾದರಿ ಸಿಗಲಿಲ್ಲ\npython train.py ಚಲಾಯಿಸಿ",
        "hero_title":  "🌾 ಕ್ಲೋರೋಫಿಲ್ SPAD ಮುನ್ಸೂಚನೆ",
        "hero_sub":    "ಎಲೆ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ · AI ಕ್ಲೋರೋಫಿಲ್ ವಿಶ್ಲೇಷಿಸುತ್ತದೆ · ತಕ್ಷಣ ಸಲಹೆ ಪಡೆಯಿರಿ",
        "upload_head": "ಎಲೆ ಚಿತ್ರ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "upload_hint": "JPG · PNG · BMP ಬೆಂಬಲಿತ",
        "tips_head":   "📌 ಫೋಟೋ ಸಲಹೆಗಳು",
        "tips":        ["ಬಿಳಿ ಅಥವಾ ಸಾದಾ ಹಿನ್ನೆಲೆ", "ನೈಸರ್ಗಿಕ ಬೆಳಕು, ನೆರಳು ಬೇಡ", "ಎಲೆಯಿಂದ 20–30 ಸೆಮೀ ದೂರ", "ಸಂಪೂರ್ಣ ಎಲೆ ಕಾಣಿಸಬೇಕು"],
        "await_title": "ಎಲೆ ಚಿತ್ರಕ್ಕಾಗಿ ಕಾಯುತ್ತಿದ್ದೇವೆ",
        "await_sub":   "ವಿಶ್ಲೇಷಣೆ ಪ್ರಾರಂಭಿಸಲು ಎಡಭಾಗದಲ್ಲಿ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "analysing":   "ಎಲೆ ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ …",
        "spad_label":  "ಅಂದಾಜು SPAD ಮೌಲ್ಯ",
        "spad_unit":   "ಕ್ಲೋರೋಫಿಲ್ ಸೂಚ್ಯಂಕ · SPAD ಘಟಕಗಳು",
        "health_lbl":  "ಎಲೆ ಆರೋಗ್ಯ",
        "n_lbl":       "ಅಂದಾಜು ಸಾರಜನಕ",
        "advice_head": "ಕೃಷಿ ಶಿಫಾರಸು",
        "saved_msg":   "✓ ಡೇಟಾಸೆಟ್‌ಗೆ ಉಳಿಸಲಾಗಿದೆ",
        "status": {
            "CRITICAL":  "🔴 ತೀವ್ರ ಸಾರಜನಕ ಕೊರತೆ. ತಕ್ಷಣ 30 ಕಿ.ಗ್ರಾ N/ಹೆ ಯೂರಿಯಾ ಹಾಕಿ.",
            "POOR":      "🟠 ಸಾರಜನಕ ಒತ್ತಡ. 20 ಕಿ.ಗ್ರಾ N/ಹೆ ವಿಂಗಡಿಸಿ ಹಾಕಿ.",
            "MODERATE":  "🟡 ಸಾಕಾಗದ ಮಟ್ಟ. 10–15 ಕಿ.ಗ್ರಾ N/ಹೆ ಹಗುರವಾಗಿ ಹಾಕಿ.",
            "GOOD":      "🟢 ಆರೋಗ್ಯಕರ ಕ್ಲೋರೋಫಿಲ್. ಪ್ರಸ್ತುತ ವೇಳಾಪಟ್ಟಿ ಮುಂದುವರೆಸಿ.",
            "EXCELLENT": "💚 ಹೆಚ್ಚಿನ ಕ್ಲೋರೋಫಿಲ್. ಶಕ್ತಿಶಾಲಿ ಬೆಳೆ. ಸಾರಜನಕ ಬೇಡ.",
        },
        "guide_title": "📖 ರೈತರ SPAD ಮಾರ್ಗದರ್ಶಿ",
        "guide_sub":   "ಕ್ಲೋರೋಫಿಲ್ ವಿವರಣೆ · ಸಾರಜನಕ ನಿರ್ವಹಣೆ · ಉತ್ತಮ ಅಭ್ಯಾಸಗಳು",
        "table_headers": ["SPAD ವ್ಯಾಪ್ತಿ","ಸ್ಥಿತಿ","ಎಲೆ ಬಣ್ಣ","ಕಾರಣ","ಕ್ರಮ"],
        "table_rows": [
            ["0–15","🔴 ತೀವ್ರ","ತೀವ್ರ ಹಳದಿ","ತೀವ್ರ N ಕೊರತೆ","30 ಕಿ.ಗ್ರಾ N/ಹೆ ತಕ್ಷಣ"],
            ["15–25","🟠 ಕಳಪೆ","ತಿಳಿ ಹಳದಿ-ಹಸಿರು","N ಕೊರತೆ","20 ಕಿ.ಗ್ರಾ N/ಹೆ + pH ತಪಾಸಣೆ"],
            ["25–35","🟡 ಮಧ್ಯಮ","ತಿಳಿ ಹಸಿರು","ಸಾಕಾಗದ","ಹಗುರ, ವಾರದಲ್ಲಿ ಮೇಲ್ವಿಚಾರಣೆ"],
            ["35–45","🟢 ಉತ್ತಮ","ಸಾಮಾನ್ಯ ಹಸಿರು","ಆರೋಗ್ಯಕರ","ವೇಳಾಪಟ್ಟಿ ಮುಂದುವರೆಸಿ"],
            ["45–80","💚 ಅತ್ಯುತ್ತಮ","ಗಾಢ ಹಸಿರು","ಹೆಚ್ಚಿನ ಕ್ಲೋರೋಫಿಲ್","ಕ್ರಮ ಬೇಡ"],
        ],
        "tips_timing": ["ಬೆಳಿಗ್ಗೆ 7–10 ಗಂಟೆಗೆ ಅಳೆಯಿರಿ","ಮೇಲಿನ ಸಂಪೂರ್ಣ ಎಲೆ ಬಳಸಿ","3 ರೀಡಿಂಗ್ ತೆಗೆದು ಸರಾಸರಿ ಲೆಕ್ಕಿಸಿ","ಪ್ರತಿ ಬಾರಿ ಅದೇ ಸ್ಥಳದಲ್ಲಿ ಅಳೆಯಿರಿ"],
        "tips_stage":  ["ಕಂದು ಹಂತ: 35–40 SPAD","ತೆನೆ ಹಂತ: 38–42 SPAD","ಶಿರ ಹಂತ: 35–38 SPAD"],
        "tips_n":      ["ಯೂರಿಯಾ 20–25 ಕಿ.ಗ್ರಾ/ಹೆ ಹಾಕಿ","2 ಕಂತುಗಳಲ್ಲಿ, 1 ವಾರ ಅಂತರ","7 ದಿನಗಳ ನಂತರ ಮತ್ತೆ ಅಳೆಯಿರಿ","ಹಾಕಿದ ನಂತರ ನೀರು ಕೊಡಿ"],
        "tips_warn":   ["SPAD > 50 → ಬೀಳುವ ಅಪಾಯ","ಹೆಚ್ಚಿನ N → ಕಂದು ಕೀಟ","ಬ್ಲಾಸ್ಟ್ ರೋಗ ಬರಬಹುದು","ಮಣ್ಣು ಪರೀಕ್ಷೆಯಿಂದ ದೃಢೀಕರಿಸಿ"],
        "farmer_quotes": [
            '"ಆರೋಗ್ಯಕರ ಎಲೆ ಎಂದರೆ ಸಮೃದ್ಧ ಫಸಲು." — ಕೃಷಿ ಜ್ಞಾನ',
            '"ಮಣ್ಣು ಕೃಷಿಯ ಅಡಿಪಾಯ, ಕ್ಲೋರೋಫಿಲ್ ಅದರ ಕನ್ನಡಿ."',
            '"ಮೊದಲು ಅಳೆಯಿರಿ, ನಂತರ ಜಾಣ್ಮೆಯಿಂದ ಗೊಬ್ಬರ ಹಾಕಿ."',
        ],
        "did_you_know": [
            "ಭತ್ತ ವಿಶ್ವದ 20% ಆಹಾರ ಶಕ್ತಿಯನ್ನು ಒದಗಿಸುತ್ತದೆ.",
            "SPAD 1 ಘಟಕ ಕಡಿಮೆಯಾದರೆ ಇಳುವರಿ 1.5% ಕಡಿಮೆಯಾಗಬಹುದು.",
            "ಸರಿಯಾದ ಸಾರಜನಕ ನಿರ್ವಹಣೆ 30–40% ಗೊಬ್ಬರ ಉಳಿಸುತ್ತದೆ.",
            "ಭಾರತ ಚೀನಾ ನಂತರ ಎರಡನೇ ಅತಿ ದೊಡ್ಡ ಭತ್ತ ಉತ್ಪಾದಕ.",
        ],
    },
}

# ── State ────────────────────────────────────────────────────────────────────
if "dark_mode"  not in st.session_state: st.session_state.dark_mode  = False
if "language"   not in st.session_state: st.session_state.language   = "English"

dark = st.session_state.dark_mode
T    = LANG[st.session_state.language]

# ── Theme ─────────────────────────────────────────────────────────────────────
if dark:
    BG=("#0e1410"); CB=("#141f14"); CBR=("#1e3a1e"); TP=("#e0f0d0"); TS=("#7aaa5a"); TM=("#4a7a3a"); GA=("#52c052"); PB=("#0e1410"); HO=("rgba(14,20,16,0.7)")
else:
    BG=("#f5f7f2"); CB=("#ffffff"); CBR=("#cde0ba"); TP=("#1a2e14"); TS=("#2d5a1e"); TM=("#5a7a4a"); GA=("#3a7a28"); PB=("#ffffff"); HO=("rgba(20,40,16,0.55)")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Kill the keyboard_double_arrow text ── */
button[data-testid="baseButton-headerNoPadding"],
button[data-testid="baseButton-header"],
[data-testid="collapsedControl"],
button[kind="header"],
.st-emotion-cache-dvne4q,
.st-emotion-cache-1rtdyuf,
[aria-label="Close sidebar"],
[aria-label="Open sidebar"] {{ display:none !important; }}

html,body,.stApp {{ background:{BG} !important; font-family:'Inter',sans-serif; }}
.block-container {{ padding:0 2rem 4rem !important; max-width:1380px !important; }}
#MainMenu,footer {{ visibility:hidden; }}

p,span,div,label,li {{ color:{TP} !important; font-family:'Inter',sans-serif !important; }}
h1,h2,h3,h4 {{ font-family:'Merriweather',serif !important; color:{TP} !important; }}

section[data-testid="stSidebar"] {{
    background:linear-gradient(180deg,#1a3a16 0%,#254d1e 60%,#1a3318 100%) !important;
    min-width:240px !important;
}}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stRadio label p {{
    color:#d8f0c0 !important; font-size:0.91rem !important;
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{ color:#a8e080 !important; }}
section[data-testid="stSidebar"] [data-testid="stRadio"] label {{
    color:#d8f0c0 !important; font-weight:500 !important; padding:3px 0 !important;
}}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{ color:#fff !important; }}
section[data-testid="stSidebar"] [data-testid="stToggle"] p {{ color:#d8f0c0 !important; }}
section[data-testid="stSidebar"] small {{ color:#6a9a50 !important; }}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] label {{ color:#d8f0c0 !important; }}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] {{
    background:#1a3a16 !important; border-color:#3a7a2a !important;
}}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] span {{ color:#d8f0c0 !important; }}

.card {{ background:{CB}; border:1px solid {CBR}; border-radius:16px; padding:22px 26px; margin-bottom:14px; box-shadow:0 2px 10px rgba(0,0,0,0.07); transition:all 0.22s; }}
.card:hover {{ transform:translateY(-2px); box-shadow:0 8px 24px rgba(0,0,0,0.12); border-color:{GA}; }}
.card-green {{ background:{"#0f2a0f" if dark else "#edf7e5"}; border:1px solid {"#2a5a2a" if dark else "#b8dca0"}; border-radius:16px; padding:22px 26px; margin-bottom:14px; transition:all 0.22s; }}
.card-green:hover {{ transform:translateY(-2px); box-shadow:0 6px 20px rgba(0,0,0,0.1); }}
.card-warn {{ background:{"#1e1400" if dark else "#fffbee"}; border:1px solid {"#4a3800" if dark else "#e0c860"}; border-radius:16px; padding:22px 26px; margin-bottom:14px; }}
.card-dark {{ background:{"#0a1a0a" if dark else "#1a3a14"}; border:1px solid {"#1e4a1e" if dark else "#2a5a20"}; border-radius:16px; padding:24px 28px; margin-bottom:14px; }}

.sc {{ background:{CB}; border:1px solid {CBR}; border-radius:14px; padding:18px 20px; margin-bottom:12px; transition:all 0.2s; cursor:default; position:relative; }}
.sc:hover {{ transform:translateY(-2px); box-shadow:0 6px 18px rgba(0,0,0,0.12); border-color:{GA}; }}
.sc-v {{ font-family:'Merriweather',serif; font-size:2rem; font-weight:700; color:{GA} !important; line-height:1.1; }}
.sc-l {{ font-family:'JetBrains Mono',monospace; font-size:0.62rem; letter-spacing:0.13em; color:{TM} !important; text-transform:uppercase; margin-top:5px; }}

.tw {{ position:relative; display:block; width:100%; }}
.tw .tip {{ visibility:hidden; opacity:0; background:#1a3a14; color:#b0e880 !important; font-family:'JetBrains Mono',monospace; font-size:0.7rem; border-radius:8px; padding:7px 12px; position:absolute; bottom:105%; left:50%; transform:translateX(-50%); white-space:nowrap; z-index:9999; transition:opacity 0.18s; box-shadow:0 4px 14px rgba(0,0,0,0.25); pointer-events:none; }}
.tw .tip::after {{ content:''; position:absolute; top:100%; left:50%; transform:translateX(-50%); border:5px solid transparent; border-top-color:#1a3a14; }}
.tw:hover .tip {{ visibility:visible; opacity:1; }}

.page-hero {{ background:linear-gradient(135deg,#1a3a14 0%,#2a5a20 55%,#356028 100%); border-radius:0 0 24px 24px; padding:32px 44px 28px; margin:0 -2rem 2rem; display:flex; align-items:center; justify-content:space-between; position:relative; overflow:hidden; }}
.page-hero::before {{ content:''; position:absolute; top:0; left:0; right:0; bottom:0; background:url('https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=1400&q=20') center/cover no-repeat; opacity:0.1; pointer-events:none; }}
.hero-title {{ font-family:'Merriweather',serif !important; font-size:1.85rem; font-weight:700; color:#fff !important; line-height:1.25; margin:0; position:relative; z-index:1; }}
.hero-sub {{ font-size:0.85rem; color:#a0d070 !important; margin-top:7px; letter-spacing:0.03em; position:relative; z-index:1; }}
.hero-badge {{ background:rgba(255,255,255,0.1); border:1px solid rgba(255,255,255,0.22); border-radius:12px; padding:10px 16px; font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#c0f088 !important; letter-spacing:0.08em; text-align:center; position:relative; z-index:1; }}

.spad-result {{ background:linear-gradient(135deg,#1a3a14 0%,#2a5a20 60%,#356028 100%); border-radius:20px; padding:34px 40px; text-align:center; box-shadow:0 8px 28px rgba(0,0,0,0.25); }}
.spad-eye {{ font-family:'JetBrains Mono',monospace; font-size:0.66rem; letter-spacing:0.22em; color:#80b860 !important; text-transform:uppercase; margin-bottom:8px; }}
.spad-num {{ font-family:'Merriweather',serif; font-size:5.5rem; font-weight:700; line-height:1; color:#c8f088 !important; text-shadow:0 2px 18px rgba(180,240,100,0.28); }}
.spad-unit {{ font-family:'JetBrains Mono',monospace; font-size:0.72rem; letter-spacing:0.15em; color:#60903a !important; margin-top:8px; }}

.ha-c {{ background:{"#1e0000" if dark else "#fff3f3"}; border-left:4px solid #e53935; border-radius:12px; padding:16px 20px; margin:10px 0; }}
.ha-p {{ background:{"#1e1000" if dark else "#fff8ee"}; border-left:4px solid #fb8c00; border-radius:12px; padding:16px 20px; margin:10px 0; }}
.ha-m {{ background:{"#1e1a00" if dark else "#fffde5"}; border-left:4px solid #f9a825; border-radius:12px; padding:16px 20px; margin:10px 0; }}
.ha-g {{ background:{"#001e00" if dark else "#eef8e8"}; border-left:4px solid #43a047; border-radius:12px; padding:16px 20px; margin:10px 0; }}
.ha-e {{ background:{"#001e10" if dark else "#e6f8ef"}; border-left:4px solid #00897b; border-radius:12px; padding:16px 20px; margin:10px 0; }}
.ha-c p,.ha-c strong {{ color:{"#ffaaaa" if dark else "#8b0000"} !important; }}
.ha-p p,.ha-p strong {{ color:{"#ffcc88" if dark else "#7a3800"} !important; }}
.ha-m p,.ha-m strong {{ color:{"#ffee88" if dark else "#6a4a00"} !important; }}
.ha-g p,.ha-g strong {{ color:{"#88ee88" if dark else "#1a4a10"} !important; }}
.ha-e p,.ha-e strong {{ color:{"#88eedd" if dark else "#003d30"} !important; }}

.fc {{ background:{"#0f2a0f" if dark else "#edf7e5"}; border:1px solid {"#2a5a2a" if dark else "#b8dca0"}; border-radius:10px; padding:14px 16px; margin-bottom:10px; transition:all 0.18s; }}
.fc:hover {{ transform:translateY(-1px); box-shadow:0 4px 12px rgba(0,0,0,0.1); }}
.fc-v {{ font-family:'Merriweather',serif; font-size:1.25rem; font-weight:700; color:{GA} !important; }}
.fc-l {{ font-family:'JetBrains Mono',monospace; font-size:0.58rem; letter-spacing:0.12em; color:{TM} !important; text-transform:uppercase; margin-top:4px; }}

.sec-head {{ font-family:'Merriweather',serif !important; font-size:1.2rem; font-weight:700; color:{TP} !important; border-left:4px solid {GA}; padding-left:14px; margin:22px 0 14px; }}
.sec-tag {{ display:inline-block; background:{"#0a2a0a" if dark else "#e0f0d4"}; border:1px solid {"#2a5a2a" if dark else "#90c870"}; color:{GA} !important; border-radius:20px; padding:3px 12px; font-size:0.7rem; font-family:'JetBrains Mono',monospace; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:10px; }}

.fim {{ border-radius:14px; overflow:hidden; box-shadow:0 4px 14px rgba(0,0,0,0.13); position:relative; margin-bottom:14px; transition:all 0.28s; }}
.fim:hover {{ transform:scale(1.02); box-shadow:0 10px 28px rgba(0,0,0,0.2); }}
.fim-cap {{ position:absolute; bottom:0; left:0; right:0; background:linear-gradient(transparent,rgba(0,0,0,0.72)); padding:18px 14px 11px; color:#fff !important; font-size:0.82rem; font-weight:500; }}

/* Quote card */
.quote-card {{ background:linear-gradient(135deg,{"#0d2a10" if dark else "#e8f5dc"},{"#1a4a1a" if dark else "#d0ecc0"}); border:1px solid {"#2a5a2a" if dark else "#a0d080"}; border-radius:14px; padding:20px 24px; margin:10px 0; border-left:4px solid {GA}; }}
.quote-text {{ font-family:'Merriweather',serif; font-style:italic; font-size:0.95rem; color:{TS} !important; line-height:1.7; }}

/* Did you know */
.dyk {{ background:{"#0a1e0a" if dark else "#f0f8e8"}; border:1px solid {CBR}; border-radius:12px; padding:14px 18px; margin:8px 0; display:flex; align-items:flex-start; gap:12px; }}
.dyk-icon {{ font-size:1.4rem; flex-shrink:0; margin-top:2px; }}
.dyk-text {{ color:{TS} !important; font-size:0.87rem; line-height:1.6; }}

[data-testid="stFileUploadDropzone"] {{ background:{CB} !important; border:2px dashed {"#3a7a28" if dark else "#8ac860"} !important; border-radius:14px !important; }}
[data-testid="stFileUploadDropzone"] p {{ color:{TS} !important; }}
.stButton>button {{ background:linear-gradient(135deg,#2a5a1e,#3a7028) !important; color:#fff !important; border:none !important; border-radius:10px !important; font-weight:600 !important; box-shadow:0 4px 14px rgba(42,90,30,0.3) !important; transition:all 0.2s !important; }}
.stButton>button:hover {{ background:linear-gradient(135deg,#1a3a14,#2a5a1e) !important; transform:translateY(-1px) !important; }}
.stSuccess {{ background:{"#001e00" if dark else "#eef8e8"} !important; border-color:#4caf50 !important; }}
.stSuccess p {{ color:{"#88ee88" if dark else "#1a4a10"} !important; }}
.stError {{ background:{"#1e0000" if dark else "#fff3f3"} !important; }}
[data-testid="stDataFrame"] {{ background:{CB} !important; border:1px solid {CBR} !important; border-radius:12px !important; }}
table {{ width:100%; border-collapse:collapse; }}
th {{ background:{"#1a3a14" if dark else "#2a5a1e"} !important; color:#c0f088 !important; font-family:'JetBrains Mono',monospace !important; font-size:0.68rem !important; letter-spacing:0.1em !important; text-transform:uppercase !important; padding:11px 15px !important; }}
td {{ color:{TP} !important; padding:10px 15px !important; border-bottom:1px solid {CBR} !important; font-size:0.87rem !important; }}
tr:hover td {{ background:{"#0f2a0f" if dark else "#f0f8e8"} !important; }}
.badge-ok {{ background:{"#001e00" if dark else "#e8f5e4"}; border:1px solid #4caf50; border-radius:9px; padding:11px 14px; font-family:'JetBrains Mono',monospace; font-size:0.76rem; color:{"#88ee88" if dark else "#1a4a10"} !important; text-align:center; }}
.badge-err {{ background:{"#1e0000" if dark else "#fff3f3"}; border:1px solid #e53935; border-radius:9px; padding:11px 14px; font-family:'JetBrains Mono',monospace; font-size:0.76rem; color:{"#ffaaaa" if dark else "#8b0000"} !important; text-align:center; }}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def model_ready():
    return os.path.exists(os.path.join("saved_model","hybrid_cnn_final.keras")) and \
           os.path.exists(os.path.join("saved_model","scaler.pkl"))

STATUS_KEYS = ["CRITICAL","POOR","MODERATE","GOOD","EXCELLENT"]
STATUS_COLORS = {"CRITICAL":"#e53935","POOR":"#fb8c00","MODERATE":"#f9a825","GOOD":"#43a047","EXCELLENT":"#00897b"}
STATUS_CLS    = {"CRITICAL":"ha-c","POOR":"ha-p","MODERATE":"ha-m","GOOD":"ha-g","EXCELLENT":"ha-e"}

def spad_key(v):
    if v<15: return "CRITICAL"
    elif v<25: return "POOR"
    elif v<35: return "MODERATE"
    elif v<45: return "GOOD"
    else: return "EXCELLENT"

def n_est(v): return f"{max(0.0,0.065*v+0.5):.2f}% N"

def tip_card(val, label, tip):
    return f"""<div class="tw"><div class="sc">
        <div class="sc-v">{val}</div><div class="sc-l">{label}</div>
    </div><span class="tip">{tip}</span></div>"""

def hdiv():
    st.markdown(f'<hr style="border:none;border-top:1px solid {CBR};margin:18px 0">', unsafe_allow_html=True)

def farm_img(url, caption, height=160):
    return f"""<div class="fim"><img src="{url}" style="width:100%;height:{height}px;object-fit:cover;display:block"/>
    <div class="fim-cap">{caption}</div></div>"""


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="padding:20px 16px 16px;border-bottom:1px solid rgba(255,255,255,0.1);margin-bottom:8px">
        <div style="font-family:'Merriweather',serif;font-size:1.35rem;font-weight:700;color:#fff">🌾 {T['app_name']}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:#6aaa40;letter-spacing:.15em;margin-top:4px">{T['tagline']}</div>
        <div style="margin-top:12px;border-radius:8px;overflow:hidden">
            <img src="https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=400&q=60"
                 style="width:100%;height:85px;object-fit:cover;display:block;border-radius:8px;opacity:0.75"/>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"**{T['nav_label']}**")
    page = st.radio("page", options=[T["nav_predict"],T["nav_dataset"],T["nav_guide"],T["nav_about"]], label_visibility="collapsed")

    st.markdown("---")
    st.markdown(f"**{T['settings']}**")

    # Language selector
    lang_choice = st.selectbox(T["language"], options=list(LANG.keys()),
                                index=list(LANG.keys()).index(st.session_state.language),
                                label_visibility="visible")
    if lang_choice != st.session_state.language:
        st.session_state.language = lang_choice
        st.rerun()

    dark_toggle = st.toggle(T["dark_mode"], value=st.session_state.dark_mode)
    if dark_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_toggle
        st.rerun()

    auto_save  = st.toggle(T["auto_save"],  value=True)
    show_feats = st.toggle(T["show_feats"], value=True)

    st.markdown("---")
    if model_ready():
        st.markdown(f'<div class="badge-ok">{T["model_ok"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="badge-err">{T["model_err"]}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""<div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:#4a7a30;letter-spacing:.07em;line-height:1.9">
    B.S. ABDUR RAHMAN CRESCENT<br>INSTITUTE OF SCIENCE & TECH<br>B.TECH ECE · APRIL 2026</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if page == T["nav_predict"]:
    st.markdown(f"""<div class="page-hero">
        <div><div class="hero-title">{T['hero_title']}</div>
        <div class="hero-sub">{T['hero_sub']}</div></div>
        <div class="hero-badge">HYBRID CNN + RIDGE ENSEMBLE<br>━━━━━━━━━━━━━<br>Accuracy 80.84% · MAE ~5 SPAD</div>
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([1,1], gap="large")

    with col_l:
        st.markdown(f'<div class="sec-head">{T["upload_head"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:{TM};font-size:.87rem;margin-bottom:10px">{T["upload_hint"]}</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["jpg","jpeg","png","bmp"], label_visibility="collapsed")

        if uploaded:
            pil = Image.open(uploaded).convert("RGB")
            st.image(pil, use_container_width=True, caption=f"📁 {uploaded.name}")
            tips_html = "".join([f'<strong style="color:{TP}">✓</strong> {t}<br>' for t in T["tips"]])
            st.markdown(f"""<div class="card-green"><div class="sec-tag">{T['tips_head']}</div>
            <p style="color:{TS};margin:10px 0 0;font-size:.87rem;line-height:1.9">{tips_html}</p></div>""", unsafe_allow_html=True)
        else:
            st.markdown(farm_img("https://images.unsplash.com/photo-1574943320219-553eb213f72d?w=600&q=70","🌾 Rice Field Monitoring", 200) +
                f'<div style="padding:14px 18px;background:{CB};border:1px solid {CBR};border-radius:0 0 16px 16px;margin-top:-6px">'
                f'<div class="sec-tag">RICE FIELD MONITORING</div>'
                f'<p style="color:{TS};font-size:.85rem;margin:8px 0 0">Upload a leaf photo to instantly measure chlorophyll content and get fertilizer recommendations.</p></div>', unsafe_allow_html=True)
            st.markdown(farm_img("https://images.unsplash.com/photo-1559827291-72ee739d0d9a?w=600&q=70","👩‍🌾 AI-Powered Crop Health", 160) +
                f'<div style="padding:12px 18px;background:{CB};border:1px solid {CBR};border-radius:0 0 16px 16px;margin-top:-6px">'
                f'<div class="sec-tag">PRECISION AGRICULTURE</div>'
                f'<p style="color:{TS};font-size:.84rem;margin:8px 0 0">Commercial SPAD-502 accuracy at a fraction of the cost — accessible to every farmer.</p></div>', unsafe_allow_html=True)

            # Did you know
            st.markdown(f'<div class="sec-head" style="font-size:1rem">💡 Did You Know?</div>', unsafe_allow_html=True)
            for fact in T["did_you_know"]:
                st.markdown(f'<div class="dyk"><div class="dyk-icon">🌱</div><div class="dyk-text">{fact}</div></div>', unsafe_allow_html=True)

    with col_r:
        if uploaded:
            if not model_ready():
                st.error(T["model_err"])
            else:
                with st.spinner(T["analysing"]):
                    img_np  = np.array(pil)
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    from predictor import predict_spad, save_new_sample
                    result  = predict_spad(img_bgr)

                spad   = result["spad_predicted"]
                sk     = spad_key(spad)
                color  = STATUS_COLORS[sk]
                cls    = STATUS_CLS[sk]
                advice = T["status"][sk]

                st.markdown(f"""<div class="spad-result">
                    <div class="spad-eye">{T['spad_label']}</div>
                    <div class="spad-num">{spad}</div>
                    <div class="spad-unit">{T['spad_unit']}</div>
                </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                c1,c2 = st.columns(2)
                with c1: st.markdown(tip_card(f'<span style="color:{color}">{sk}</span>',T["health_lbl"],f"SPAD={spad}"), unsafe_allow_html=True)
                with c2: st.markdown(tip_card(n_est(spad),T["n_lbl"],"Empirical N estimate"), unsafe_allow_html=True)

                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=spad,
                    number={"font":{"size":26,"color":GA,"family":"Merriweather"},"suffix":" SPAD"},
                    gauge={"axis":{"range":[0,80],"tickcolor":TM,"tickfont":{"color":TM,"size":9}},
                           "bar":{"color":color,"thickness":0.22},"bgcolor":CB,"bordercolor":CBR,
                           "steps":[{"range":[0,15],"color":"#3a1010" if dark else "#ffe5e5"},
                                    {"range":[15,25],"color":"#3a2010" if dark else "#fff3e0"},
                                    {"range":[25,35],"color":"#3a3010" if dark else "#fffde7"},
                                    {"range":[35,45],"color":"#103a10" if dark else "#f1f8e9"},
                                    {"range":[45,80],"color":"#103a28" if dark else "#e0f4f0"}],
                           "threshold":{"line":{"color":TP,"width":2},"value":spad,"thickness":0.85}},
                    title={"text":"SPAD HEALTH GAUGE","font":{"color":TM,"size":10,"family":"JetBrains Mono"}}
                ))
                fig.update_layout(height=210,paper_bgcolor=PB,margin=dict(t=42,b=0,l=18,r=18))
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(f'<div class="{cls}"><strong>{T["advice_head"]}</strong><p style="margin-top:8px;font-size:.88rem">{advice}</p></div>', unsafe_allow_html=True)

                if show_feats:
                    st.markdown(f'<div class="sec-head" style="font-size:.95rem;margin-top:16px">🎨 {T["show_feats"]}</div>', unsafe_allow_html=True)
                    feats = result["color_features"]
                    tip_map={"Norm-R":"Low in healthy green leaves","Norm-G":"Higher = more chlorophyll","Norm-B":"Absorbed by chlorophyll","Hue":"Leaf hue angle","Saturation":"Colour intensity","Value":"Pixel brightness","DGCI":"Higher = healthier leaf"}
                    c4 = st.columns(4)
                    for i,(n,v) in enumerate(feats.items()):
                        with c4[i%4]:
                            st.markdown(f'<div class="tw"><div class="fc"><div class="fc-v">{v:.4f}</div><div class="fc-l">{n}</div></div><span class="tip">{tip_map.get(n,"")}</span></div>', unsafe_allow_html=True)

                    fv,fk=list(feats.values()),list(feats.keys())
                    fr=go.Figure(go.Scatterpolar(r=fv+[fv[0]],theta=fk+[fk[0]],fill="toself",fillcolor=f"rgba(74,140,50,{'0.15' if dark else '0.1'})",line=dict(color=GA,width=2),marker=dict(color=GA,size=6)))
                    fr.update_layout(polar=dict(bgcolor=CB,radialaxis=dict(visible=True,range=[0,1],gridcolor=CBR,tickfont={"color":TM,"size":8}),angularaxis=dict(gridcolor=CBR,tickfont={"color":TS,"size":10})),showlegend=False,title={"text":"COLOUR FEATURE RADAR","font":{"color":TM,"size":10,"family":"JetBrains Mono"}},height=270,paper_bgcolor=PB,margin=dict(t=42,b=8,l=28,r=28))
                    st.plotly_chart(fr, use_container_width=True)

                if auto_save:
                    saved = save_new_sample(img_bgr, uploaded.name, spad)
                    st.success(f"{T['saved_msg']}: `{saved}` | SPAD = {spad}")
                else:
                    st.markdown(f'<div class="ha-m"><strong>Auto-save OFF</strong><p>Toggle in sidebar to enable.</p></div>', unsafe_allow_html=True)
        else:
            # Farmer quotes
            st.markdown(f'<div class="sec-head">🌾 Farmer Wisdom</div>', unsafe_allow_html=True)
            for q in T["farmer_quotes"]:
                st.markdown(f'<div class="quote-card"><div class="quote-text">{q}</div></div>', unsafe_allow_html=True)

            st.markdown(f"""<div style="height:220px;display:flex;flex-direction:column;align-items:center;
                justify-content:center;background:{CB};border:2px dashed {CBR};border-radius:20px;margin-top:14px">
                <div style="font-size:3rem;margin-bottom:14px;opacity:0.6">🍃</div>
                <div style="font-family:'Merriweather',serif;font-weight:700;font-size:1rem;color:{TS}">{T['await_title']}</div>
                <div style="font-size:.82rem;color:{TM};margin-top:6px">{T['await_sub']}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
elif page == T["nav_dataset"]:
    st.markdown(f"""<div class="page-hero"><div>
        <div class="hero-title">📊 Dataset Intelligence</div>
        <div class="hero-sub">Monitor training data · Track SPAD distribution · Manage continuous learning</div>
    </div></div>""", unsafe_allow_html=True)

    CSV = "rice_labels.csv"
    if not os.path.exists(CSV):
        st.warning("Dataset file rice_labels.csv not found.")
    else:
        df   = pd.read_csv(CSV)
        sc   = next((c for c in df.columns if "spad" in c.lower()),None)
        vals = pd.to_numeric(df[sc],errors="coerce").dropna() if sc else pd.Series([])

        c1,c2,c3,c4 = st.columns(4)
        for col,lbl,val,tip in zip([c1,c2,c3,c4],
            ["TOTAL IMAGES","MEAN SPAD","SPAD RANGE","STD DEV"],
            [len(df),f"{vals.mean():.1f}" if len(vals) else "—",
             f"{vals.min():.0f}–{vals.max():.0f}" if len(vals) else "—",
             f"±{vals.std():.2f}" if len(vals) else "—"],
            ["Total leaf images in dataset","Average SPAD value","Min to Max SPAD","Variability in readings"]):
            with col: st.markdown(tip_card(val,lbl,tip), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        ct,cg = st.columns([1,1],gap="large")
        with ct:
            st.markdown(f'<div class="sec-head">Recent Entries</div>', unsafe_allow_html=True)
            st.dataframe(df.tail(15), use_container_width=True, hide_index=True)
        with cg:
            if sc and len(vals):
                fig=go.Figure(go.Histogram(x=vals,nbinsx=20,marker_color=GA,marker_line_color=BG,marker_line_width=1.5,opacity=0.85))
                fig.update_layout(title={"text":"SPAD DISTRIBUTION","font":{"color":TM,"size":11,"family":"JetBrains Mono"}},xaxis={"title":"SPAD Value","color":TS,"gridcolor":CBR},yaxis={"title":"Count","color":TS,"gridcolor":CBR},paper_bgcolor=PB,plot_bgcolor=CB,font_color=TP,height=300,margin=dict(t=42,b=40,l=40,r=20))
                st.plotly_chart(fig, use_container_width=True)

        st.markdown(f'<div class="ha-g"><strong>CONTINUOUS LEARNING</strong><p>After 10+ new auto-saved predictions, retrain:<br><code>python train.py --images rice_images --excel rice_labels.csv --epochs 150</code></p></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SPAD GUIDE
# ══════════════════════════════════════════════════════════════════════════════
elif page == T["nav_guide"]:
    st.markdown(f"""<div class="page-hero"><div>
        <div class="hero-title">{T['guide_title']}</div>
        <div class="hero-sub">{T['guide_sub']}</div>
    </div></div>""", unsafe_allow_html=True)

    # 4 farmer images
    ci1,ci2,ci3,ci4 = st.columns(4)
    for col,url,cap in zip([ci1,ci2,ci3,ci4],
        ["https://images.unsplash.com/photo-1574943320219-553eb213f72d?w=500&q=70",
         "https://images.unsplash.com/photo-1625246333195-78d9c38ad449?w=500&q=70",
         "https://images.unsplash.com/photo-1464226184884-fa280b87c399?w=500&q=70",
         "https://images.unsplash.com/photo-1559827291-72ee739d0d9a?w=500&q=70"],
        ["Rice Field Monitoring","Crop Health Assessment","Precision Fertilisation","Farmer with Crops"]):
        with col: st.markdown(farm_img(url,cap,140), unsafe_allow_html=True)

    hdiv()
    st.markdown(f'<div class="sec-head">SPAD Value Interpretation</div>', unsafe_allow_html=True)
    hdr = T["table_headers"]
    rows_html = ""
    status_colors_td={"🔴":"#e53935","🟠":"#fb8c00","🟡":"#f9a825","🟢":"#43a047","💚":"#00897b"}
    for row in T["table_rows"]:
        clr = next((v for k,v in status_colors_td.items() if k in row[1]), TP)
        rows_html += f'<tr><td><strong style="color:{TP}">{row[0]}</strong></td><td style="color:{clr};font-weight:700">{row[1]}</td>' + "".join(f'<td style="color:{TP}">{c}</td>' for c in row[2:]) + "</tr>"
    st.markdown(f'<table><tr>{"".join(f"<th>{h}</th>" for h in hdr)}</tr>{rows_html}</table>', unsafe_allow_html=True)

    hdiv()
    ca,cb = st.columns(2,gap="large")
    with ca:
        st.markdown(f'<div class="sec-head">Best Measurement Practices</div>', unsafe_allow_html=True)
        tips_t = "".join(f'• {t}<br>' for t in T["tips_timing"])
        tips_s = "".join(f'• {t}<br>' for t in T["tips_stage"])
        st.markdown(f"""<div class="card-green"><div class="sec-tag">⏰ Timing & Method</div>
        <p style="color:{TS};margin-top:10px;font-size:.89rem;line-height:1.9">{tips_t}</p></div>
        <div class="card-green"><div class="sec-tag">🌾 Growth Stage Targets</div>
        <p style="color:{TS};margin-top:10px;font-size:.89rem;line-height:1.9">{tips_s}</p></div>""", unsafe_allow_html=True)
    with cb:
        st.markdown(f'<div class="sec-head">Nitrogen Management</div>', unsafe_allow_html=True)
        tips_n = "".join(f'• {t}<br>' for t in T["tips_n"])
        tips_w = "".join(f'• {t}<br>' for t in T["tips_warn"])
        st.markdown(f"""<div class="card-green"><div class="sec-tag">💧 If SPAD Below 35</div>
        <p style="color:{TS};margin-top:10px;font-size:.89rem;line-height:1.9">{tips_n}</p></div>
        <div class="card-warn"><div class="sec-tag" style="background:{'#2a1a00' if dark else '#fff3cd'};border-color:{'#5a3800' if dark else '#e0c860'};color:{'#e0a820' if dark else '#7a5000'}">⚠ Over-Fertilisation</div>
        <p style="color:{'#e0c060' if dark else '#6d4c00'};margin-top:10px;font-size:.89rem;line-height:1.9">{tips_w}</p></div>""", unsafe_allow_html=True)

    hdiv()
    # Farmer quotes section
    st.markdown(f'<div class="sec-head">🌾 Farmer Wisdom</div>', unsafe_allow_html=True)
    qc1,qc2,qc3 = st.columns(3)
    for col,q in zip([qc1,qc2,qc3],T["farmer_quotes"]):
        with col:
            st.markdown(f'<div class="quote-card"><div class="quote-text">{q}</div></div>', unsafe_allow_html=True)

    hdiv()
    st.markdown(f'<div class="sec-head">How SPAD is Calculated</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="sec-tag">Optical Transmission Index</div>', unsafe_allow_html=True)
        st.latex(r"\text{Index}=\ln\!\left(\frac{V_{NIR}-V_{dark}}{V_{Red}-V_{dark}}\right)")
    with c2:
        st.markdown(f'<div class="sec-tag">SPAD Calibration Formula</div>', unsafe_allow_html=True)
        st.latex(r"\text{SPAD}_{HW}=A\times\text{Index}+B")
    st.markdown(f'<p style="color:{TM};font-size:.83rem">A and B calibrated against commercial SPAD-502 meter (Konica Minolta).</p>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == T["nav_about"]:
    st.markdown(f"""<div class="page-hero"><div>
        <div class="hero-title">⚙️ About LeafSense</div>
        <div class="hero-sub">Project team · System architecture · Model performance · Technology stack</div>
    </div></div>""", unsafe_allow_html=True)

    ca,cb = st.columns([3,2],gap="large")
    with ca:
        st.markdown(f"""<div class="card-green"><div class="sec-tag">PROJECT TITLE</div>
        <div style="font-family:'Merriweather',serif;font-size:1.1rem;font-weight:700;color:{TP};margin:12px 0 8px">
        Development of a Low-Cost SPAD-Based Chlorophyll Detection Device</div>
        <p style="color:{TS};font-size:.89rem">B.Tech Final Year Project · Electronics & Communication Engineering<br>
        B.S. Abdur Rahman Crescent Institute of Science & Technology · April 2026</p></div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="sec-head">Project Team</div>', unsafe_allow_html=True)
        st.markdown(f"""<table>
        <tr><th>Name</th><th>Roll No</th><th>Role</th></tr>
        <tr><td style="color:{TP}"><strong>Rahul Bhat</strong></td><td style="font-family:JetBrains Mono,monospace;color:{GA}">220051601094</td><td style="color:{TP}">Hardware + Firmware</td></tr>
        <tr><td style="color:{TP}"><strong>Y. Sai Avinash</strong></td><td style="font-family:JetBrains Mono,monospace;color:{GA}">220051601121</td><td style="color:{TP}">AI + Dataset</td></tr>
        <tr><td style="color:{TP}"><strong>Shaik Aseed</strong></td><td style="font-family:JetBrains Mono,monospace;color:{GA}">220051601103</td><td style="color:{TP}">Integration + Testing</td></tr>
        </table>
        <p style="color:{TS};margin-top:10px">Guide: <strong style="color:{TP}">Dr. C. Tharani</strong> — Professor & Dean, School of ECE</p>""", unsafe_allow_html=True)

    with cb:
        st.markdown(f'<div class="sec-head">Model Performance</div>', unsafe_allow_html=True)
        for lbl,val,tip in [("Accuracy","80.84%","Health classification"),("R² Score","0.5723","SPAD correlation"),("MAE","5.50 SPAD","Avg prediction error"),("Loss Fn","Huber","Robust to outliers"),("Optimizer","Adam","lr=0.001")]:
            st.markdown(tip_card(val,lbl,tip), unsafe_allow_html=True)

    hdiv()
    # Impact section
    st.markdown(f'<div class="sec-head">🌍 Project Impact</div>', unsafe_allow_html=True)
    imp1,imp2,imp3 = st.columns(3)
    for col,icon,title,desc in zip([imp1,imp2,imp3],
        ["💰","🌿","📱"],
        ["Cost Reduction","Sustainable Farming","Accessible Technology"],
        ["Reduces fertilizer cost by 30–40% through precision nitrogen management based on real SPAD readings.",
         "Prevents over-fertilisation, reducing environmental pollution and improving soil health.",
         "Brings commercial-grade chlorophyll sensing to every farmer with just a smartphone."]):
        with col:
            st.markdown(f"""<div class="card" style="text-align:center">
            <div style="font-size:2.5rem;margin-bottom:10px">{icon}</div>
            <div style="font-family:'Merriweather',serif;font-weight:700;font-size:1rem;color:{TP};margin-bottom:8px">{title}</div>
            <p style="font-size:.84rem;color:{TS}">{desc}</p></div>""", unsafe_allow_html=True)

    hdiv()
    cx,cy = st.columns(2,gap="large")
    with cx:
        st.markdown(f'<div class="sec-head">AI Pipeline</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="card-green"><p style="font-family:'JetBrains Mono',monospace;font-size:.78rem;color:{TS};line-height:2.1;margin:0">
        📷 Image Upload →<br>🔲 Leaf Segmentation (Otsu) →<br>🧠 CNN Feature Extraction →<br>
        🎨 Colour Features (RGB·HSV·DGCI) →<br>🔗 Feature Fusion →<br>📈 Ridge Corrector →<br>📊 SPAD + Dataset Update
        </p></div>""", unsafe_allow_html=True)
    with cy:
        st.markdown(f'<div class="sec-head">Hardware Module</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="card-green"><p style="font-family:'JetBrains Mono',monospace;font-size:.78rem;color:{TS};line-height:2.1;margin:0">
        💡 Red LED (650 nm) →<br>💡 NIR LED (940 nm) →<br>🔆 OPT101 Photodiode →<br>
        ⚡ ESP32 ADC (GPIO34) →<br>🔢 SPAD Calculation →<br>📺 OLED (SSD1306) →<br>📡 BLE Transmission
        </p></div>""", unsafe_allow_html=True)
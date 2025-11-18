# streamlit_py
import os, re
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 페이지/스타일
# ======================
st.set_page_config(page_title="Fastai 이미지 분류기", page_icon="🤖", layout="wide")
st.markdown("""
<style>
h1 { color:#1E88E5; text-align:center; font-weight:800; letter-spacing:-0.5px; }
.prediction-box { background:#E3F2FD; border:2px solid #1E88E5; border-radius:12px; padding:22px; text-align:center; margin:16px 0; box-shadow:0 4px 10px rgba(0,0,0,.06);}
.prediction-box h2 { color:#0D47A1; margin:0; font-size:2.0rem; }
.prob-card { background:#fff; border-radius:10px; padding:12px 14px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,.06); }
.prob-bar-bg { background:#ECEFF1; border-radius:6px; width:100%; height:22px; overflow:hidden; }
.prob-bar-fg { background:#4CAF50; height:100%; border-radius:6px; transition:width .5s; }
.prob-bar-fg.highlight { background:#FF6F00; }
.info-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; }
.card { border:1px solid #e3e6ea; border-radius:12px; padding:14px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,.05); }
.card h4 { margin:0 0 10px; font-size:1.05rem; color:#0D47A1; }
.thumb { width:100%; height:auto; border-radius:10px; display:block; }
.thumb-wrap { position:relative; display:block; }
.play { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:60px; height:60px; border-radius:50%; background:rgba(0,0,0,.55); }
.play:after{ content:''; border-style:solid; border-width:12px 0 12px 20px; border-color:transparent transparent transparent #fff; position:absolute; top:50%; left:50%; transform:translate(-40%,-50%); }
.helper { color:#607D8B; font-size:.9rem; }
.stFileUploader, .stCameraInput { border:2px dashed #1E88E5; border-radius:12px; padding:16px; background:#f5fafe; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 분류기 (Fastai) — 확률 막대 + 라벨별 고정 콘텐츠")

# ======================
# 세션 상태
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# 모델 로드
# ======================
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "16nuMOswv0nRdha_b1NfaqO5VyLLLdB97")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return load_learner(output_path, cpu=True)

with st.spinner("🤖 모델 로드 중..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 라벨 이름 매핑: 여기를 채우세요!
# 각 라벨당 최대 3개씩 표시됩니다.
# ======================
CONTENT_BY_LABEL: dict[str, dict[str, list[str]]] = {
    
     labels[0]: {
       "texts": ["눈은 시각 정보를 뇌에 전달하는 중요한 신체 기관으로, 카메라와 유사한 구조를 가지고 있습니다. 각막과 수정체가 빛을 굴절시켜 망막에 상을 맺게 하고, 홍채가 동공 크기를 조절하며, 망막에 있는 시세포가 빛을 전기 신호로 바꿔 시신경을 통해 뇌로 전달하는 과정을 통해 사물을 인식합니다. "],
       "images": ["https://i.namu.wiki/i/EIU8aQ93hfcrDFAV8jkmzH4eqDCkU2fyL1vSolLn6YoXN8YfTmHb9DUR7ruJd7zAPnk6vgYt1xS582rvm8qQRQ.webp"],
       "videos": ["https://www.youtube.com/watch?v=XERplfomyFs"]
     },

 labels[1]: {
       "texts": ["사람과 동물의 몸 부위 중 가장 아래쪽에 위치해 있으며, 면적에 비해 엄청난 체중을 지탱하는 기관. 고된 일을 눈에 띄지 않게 해주고 있다. 기본적으로 손과 뼈 개수 자체는 거의 같으나, 2족 보행에 알맞도록 진화가 이루어져 있다. 대표적인 것이 손바닥에서는 일부러 눌러 보지 않는 이상 눈에 잘 띄지 않는 발바닥의 오목한 부분으로, 이 아치형 구조를 통해 체중을 지탱하면서 충격을 완화시킨다. 발뒤꿈치도 마찬가지. 모든 동물군을 통틀어서 상당히 특이한 진화에 속한다."],
       "images": ["https://cdn.news.hidoc.co.kr/news/photo/201907/19665_46800_0606.jpg"],
       "videos": ["https://www.youtube.com/shorts/6J11hReO3oE"]
     },
labels[2]: {
       "texts": ["손으로도 언어처럼 자신의 의사를 표현할 수 있기 때문에 손으로도 의사를 전달할 수 있는 방법인 수어가 생기기도 했다. 또한 수어까지는 아니지만, 조롱과 경멸의 뜻을 손으로 많이 표현하는 것 역시 비슷한 맥락이라고 할 수 있다. 손을 이용해서 욕을 하는 행위 역시 전 세계적으로 발견된다. 전 세계적으로 쓰이는 가운뎃손가락을 드는 욕부터 시작해, 문화마다 매우 다양한 손가락 욕이 존재한다.

손으로 할 수 있는 일은 헤아리기 힘들 정도로 많으며, 사실 상 인간의 행동 양상중에 손으로 안되는 것부터 세는 게 빠를 정도다. 그중 손으로 할 수 있는 대표적인 일들로는 글을 쓰거나, 물건을 움직이는 물리력을 행사하거나, 그림을 그리거나, 대인관계를 형성하고 유지하거나, 식사할 때의 사용 등을 들 수 있다.

손은 인간의 지능을 직접적으로 보여주는 기관이자, 그 인간만의 고유한 특징을 드러내는 기관이다. 심지어는 인간의 나이나 건강 상태 등을 대략적으로 가늠할 수 있는 부위이기도 하다. 실제로 다 그런건 아니지만 나이가 들수록 잔뼈가 드러나고, 표면이 거칠어진다.[3] 그리고 비만도까지도 알 수 있는데, 다른 부위를 보지 않고 손만 봐도 그 사람이 살찐 정도를 상당히 정확하게 추측할 수 있다.[4] 이런 손은 주먹을 쥐면 상당히 동그랗고 짜리몽땅해 보이는데 이를 마치 만화 도라에몽에 나오는 도라에몽의 주먹과 비슷하다하여 농담조로 도라에몽 주먹 이라고도 한다.

인간은 자신이 쓰기 편한 한쪽 손만 집중적으로 더 사용하는 경향이 있다. 왼손을 주로 사용하는 경향보다는 오른손을 주로 사용하는 경향이 압도적으로 많이 보인다.

오른손잡이라도 실은 오른손과 왼손의 사용빈도가 거의 차이 나지 않는다. 오른손 사용빈도를 100이라 하면 왼손은 95 정도. 글을 쓴다거나 식사를 하는 것과 같은 섬세하고 정교한 행위들을 주로 오른손으로 하기에 체감상으론 오른손을 압도적으로 많이 쓴다고 생각하게 되지만, 별 생각 없이 무언가를 잡는다든가 짐을 운반한다든가 하는 일에 알게 모르게 왼손을 많이 쓰고 있다. 당장 오늘 아침에 몸을 씻을 때도 당신은 오른손만 쓴 게 아니라 왼손을 함께 사용했을 것이다. 또한 컴퓨터를 하거나 책을 읽을 때, 운전을 할 때 등 양손을 동원할 일은 매우 많으며 이렇듯 오른손잡이라 해도 왼손 역시 매우 중요하고 빈번하게 사용하고 있다. 왼손잡이라면 이 문단에서 오른손과 왼손을 바꿔 생각하면 된다.

손바닥은 인간의 신체 부위 중에서도 가장 빛의 영향을 덜 받는 부위 중 하나이기 때문에 멜라닌이 적다. 때문에 흑인은 손바닥만은 밝은색이고 황인이나 백인도 미세혈관이 두드러져서 붉그스름한 경우가 많다. 발바닥도 마찬가지다.

남녀 구별 없이 손이 예쁜 사람 (섬섬옥수를 가진 사람)에게 호감을 느끼는 사람도 있다.

대단히 섬세하고 중요한 부위지만 인간은 위험이 접근하면 손부터 뻗는 본능이 있어서 가장 먼저 수난을 당하는 부위가 되기 일쑤다. 소중한 손을 위험에 노출시키는 본능이 있는 이유는, 인간은 대다수 일을 손을 통해 하기에 위험에 대한 대처도 자연스레 손으로 하려는 경로의존성이 있고 이를 따른 결과도 나쁘지 않았기 때문으로 보인다. 손은 약하지만 인간의 지능과 결합하면 아무리 어려운 일도 극복할 수 있는 수단이 되기에 리스크보다 리턴이 클 확률이 높은 것. 다른 신체기관으로는 손 만큼의 대처를 하지 못해 오히려 생존에 불리했을 수 있고, 설령 손을 심각하게 다친다 해도 그 부상을 머리나 가슴에 입었으면 그냥 죽었을 확률이 높단 걸 생각하면 위험 앞에서 손을 먼저 뻗는 건 합리적인 선택이다.[5]

격투상황이 발생하면 가장 우선적으로 사용되는 부위이다. 아무래도 제일 사용하기 쉽고[6] 온갖 급소가 모여있는 얼굴[7]을 빠르게 타격하기 유리해서일 것이다. 다만 손의 내구력은 신체 부위 중 약한 축에 들기 때문에 잘못 타격할 경우 오히려 때린 쪽이 주먹에 부상을 당하기 쉽다. 따라서 제대로 싸움을 준비할 때는 손에 보호구를 끼거나 무기를 쥐게 된다.

인간이 쓰다듬어주는 행위에 중독(?)되는 동물도 상당히 많다. 동물의 다리로는 인간의 손처럼 정교하게 간지러운 곳을 긁거나 예민한 곳을 쓸기 힘들기 때문.

아기의 손 힘은 다른 부위들에 비해 빠르게 발달되는 편이다.[8]이는 영장류의 종족 특성으로, 영장류는 새끼 시절 부모의 털(사람의 경우 옷)에 매달려 지내기 때문에 나무에서 지내는 영장류가 아니더라도 다른 부위에 비해 손 힘이 강하게 발달하며, 이는 나무에 매달릴 일이 없어져 악력이 퇴화한 인류에게도 흔적 기관처럼 남은 과정이다. 때문에 인간의 악력은 유아 시기에 꽤 강하지만 점점 퇴화하다가 2차 성징 시 폭발적인 성장을 하며 차차 강해지게 되는 것."],
       "images": ["https://cdn.news.hidoc.co.kr/news/photo/201907/19665_46800_0606.jpg"],
       "videos": ["https://www.youtube.com/shorts/6J11hReO3oE"]
     },
    
}

# ======================
# 유틸
# ======================
def load_pil_from_bytes(b: bytes) -> Image.Image:
    pil = Image.open(BytesIO(b))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB": pil = pil.convert("RGB")
    return pil

def yt_id_from_url(url: str) -> str | None:
    if not url: return None
    pats = [r"(?:v=|/)([0-9A-Za-z_-]{11})(?:\?|&|/|$)", r"youtu\.be/([0-9A-Za-z_-]{11})"]
    for p in pats:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def yt_thumb(url: str) -> str | None:
    vid = yt_id_from_url(url)
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

def pick_top3(lst):
    return [x for x in lst if isinstance(x, str) and x.strip()][:3]

def get_content_for_label(label: str):
    """라벨명으로 콘텐츠 반환 (texts, images, videos). 없으면 빈 리스트."""
    cfg = CONTENT_BY_LABEL.get(label, {})
    return (
        pick_top3(cfg.get("texts", [])),
        pick_top3(cfg.get("images", [])),
        pick_top3(cfg.get("videos", [])),
    )

# ======================
# 입력(카메라/업로드)
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라로 촬영", "📁 파일 업로드"])
new_bytes = None

with tab_cam:
    cam = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if cam is not None:
        new_bytes = cam.getvalue()

with tab_file:
    f = st.file_uploader("이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
                         type=["jpg","png","jpeg","webp","tiff"])
    if f is not None:
        new_bytes = f.getvalue()

if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 예측 & 레이아웃
# ======================
if st.session_state.img_bytes:
    top_l, top_r = st.columns([1, 1], vertical_alignment="center")

    pil_img = load_pil_from_bytes(st.session_state.img_bytes)
    with top_l:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    with st.spinner("🧠 분석 중..."):
        pred, pred_idx, probs = learner.predict(PILImage.create(np.array(pil_img)))
        st.session_state.last_prediction = str(pred)

    with top_r:
        st.markdown(
            f"""
            <div class="prediction-box">
                <span style="font-size:1.0rem;color:#555;">예측 결과:</span>
                <h2>{st.session_state.last_prediction}</h2>
                <div class="helper">오른쪽 패널에서 예측 라벨의 콘텐츠가 표시됩니다.</div>
            </div>
            """, unsafe_allow_html=True
        )

    left, right = st.columns([1,1], vertical_alignment="top")

    # 왼쪽: 확률 막대
    with left:
        st.subheader("상세 예측 확률")
        prob_list = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda x: x[1], reverse=True
        )
        for lbl, p in prob_list:
            pct = p * 100
            hi = "highlight" if lbl == st.session_state.last_prediction else ""
            st.markdown(
                f"""
                <div class="prob-card">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <strong>{lbl}</strong><span>{pct:.2f}%</span>
                  </div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fg {hi}" style="width:{pct:.4f}%;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )

    # 오른쪽: 정보 패널 (예측 라벨 기본, 다른 라벨로 바꿔보기 가능)
    with right:
        st.subheader("라벨별 고정 콘텐츠")
        default_idx = labels.index(st.session_state.last_prediction) if st.session_state.last_prediction in labels else 0
        info_label = st.selectbox("표시할 라벨 선택", options=labels, index=default_idx)

        texts, images, videos = get_content_for_label(info_label)

        if not any([texts, images, videos]):
            st.info(f"라벨 `{info_label}`에 대한 콘텐츠가 아직 없습니다. 코드의 CONTENT_BY_LABEL에 추가하세요.")
        else:
            # 텍스트
            if texts:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for t in texts:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 12;">
                      <h4>텍스트</h4>
                      <div>{t}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 이미지(최대 3, 3열)
            if images:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for url in images[:3]:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 4;">
                      <h4>이미지</h4>
                      <img src="{url}" class="thumb" />
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 동영상(유튜브 썸네일)
            if videos:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for v in videos[:3]:
                    thumb = yt_thumb(v)
                    if thumb:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank" class="thumb-wrap">
                            <img src="{thumb}" class="thumb"/>
                            <div class="play"></div>
                          </a>
                          <div class="helper">{v}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank">{v}</a>
                        </div>
                        """, unsafe_allow_html=True)
else:
    st.info("카메라로 촬영하거나 파일을 업로드하면 분석 결과와 라벨별 콘텐츠가 표시됩니다.")

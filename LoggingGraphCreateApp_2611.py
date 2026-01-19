import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
from logging.handlers import RotatingFileHandler
import sys, traceback

try:
    from CTkMessagebox import CTkMessagebox
    MSG_OK = lambda t,m: CTkMessagebox(title=t, message=m)
    MSG_ERR = lambda t,m: CTkMessagebox(title=t, message=m, icon="cancel")
except ImportError:
    from tkinter import messagebox
    MSG_OK = lambda t,m: messagebox.showinfo(t,m)
    MSG_ERR = lambda t,m: messagebox.showerror(t,m)

# ================================
# ログ設定
# ================================
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

log = logging.getLogger("csv_app_logger")
log.setLevel(logging.INFO)

handler = RotatingFileHandler(LOG_DIR/"app.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
log.addHandler(handler)


# ================================
# ▼ 未処理例外→ログ自動保存（最重要！）
# ================================
def excepthook(exc_type, exc_value, exc_traceback):
    error_text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    log.error("\n====== 未処理例外発生 ======\n" + error_text)

sys.excepthook = excepthook


# ================================
# ▼ print() もログへ送る (stderr含む)
# ================================
class LogRedirector:
    def write(self, msg):
        msg = msg.strip()
        if msg:
            log.info(msg)
    def flush(self):
        pass

sys.stdout = LogRedirector()
sys.stderr = LogRedirector()


# ================================
# JSON config
# ================================
CONFIG_FILE = "config.json"

def load_config():
    if Path(CONFIG_FILE).exists():
        try:
            return json.load(open(CONFIG_FILE, "r", encoding="utf-8"))
        except:
            return {}
    return {}

def save_config(data:dict):
    json.dump(data, open(CONFIG_FILE,"w",encoding="utf-8"), indent=2)


def safe_read_csv(path, **kw):
    """
    sep / header / skiprows / nrows など既存引数を維持したまま、
    UTF-8 / UTF-16 / Shift-JIS / CP932 を順番に試す。
    """

    encodings = [
        "utf-8-sig",
        "utf-8",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "shift_jis",
        "cp932",
    ]

    # pandas に渡す共通パラメータ
    sep      = kw.get("sep", None)
    header   = kw.get("header", "infer")
    skiprows = kw.get("skiprows", None)
    nrows    = kw.get("nrows", None)

    params = {
        "sep": sep,
        "header": header,
        "skiprows": skiprows,
        "nrows": nrows,
        "engine": "python",
        "on_bad_lines": "skip",
    }

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **params)
        except Exception:
            continue

    # 最後の fallback
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace", **params)

# ------------------------------------------------
# ロギング周期予測
# ------------------------------------------------
def estimate_perios_seconds(dt_series: pd.Series) -> float:
    s = pd.to_datetime(dt_series, error="coerce").dropna().sort_values()
    if len(s) < 5:
        return 1.0
    diffs = s.diff().dt.total_seconds().dropna()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 1.0
    med = float(diffs.median())
    diffs = diffs[(diffs >= med * 0.2) & (diffs <= med * 5.0)]
    return float(diffs2.median()) if not diffs2.empty else med

# ------------------------------------------------
# サンプル番号取得、ずれすぎた点の削除
# ------------------------------------------------
def add_sample_index(
    df: pd.DataFrame,
    time_col: str,
    t_ref: pd.Timestamp,
    period_s: float,
    rounding: str = "round",
    tolerance_s: float | None = None
) -> pd.DataFrame:
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col])

    kf = (d[time_col] - t_ref).dt.total_seconds() / float(period_s)

    if rounding == "floor":
        k = np.floor(kf)
    elif rounding == "ceil":
        k = np.ceil(kf)
    else:
        k = np.rint(kf)

    if tolerance_s is not None:
        ideal = t_ref + pd.to_timedelta(k * float(period_s), unit="s")
        err= (d[time_col] - ideal).dt.total_seconds().abs()
        k = k.astype("float")
        k[err > float(tolerance_s)] = np.nan

    d["__k__"] = pd.Series(k, index=d.index).astype("Int64")
    d = d.dropna(subset=["__k__"])
    return d

# ------------------------------------------------
# CSV/TXT読込
# ------------------------------------------------
def read_data_auto(path):
    return safe_read_csv(path)


# ------------------------------------------------
# 開始/終了取得
# ------------------------------------------------
def get_start_end(path, sep, data_start_line):
    enc = ["utf-8","utf-16","utf-16-le","utf-16-be"]
    lines = None
    for e in enc:
        try:
            with open(path,"r",encoding=e) as f:
                lines=f.readlines()
            break
        except:
            continue

    if lines is None:
        return "??","??"

    start_line_idx = max(data_start_line -1, 0)
    start = "??"
    for i in range(start_line_idx, len(lines)):
        l = lines[i].strip()
        if not l:
            continue
        parts = l.split(sep)
        if parts:
            start = parts[0].strip()
            break

    end = "??"
    for l in reversed(lines):
        l = l.strip()
        if not l:
            continue
        parts = l.split(sep)
        if parts:
            end = parts[0].strip()
            break
        
    return start,end

# ------------------------------------------------
# ヘッダファイル（1行CSV/TXT）からヘッダ名リスト取得
# ------------------------------------------------
def read_header_line(path, sep=None):
    try:
        df = safe_read_csv(path, header=None, nrows=1)
        if df is None or df.empty:
            return None
        
        cols = df.iloc[0].tolist()

        cols = [str(c).strip() for c in cols]
        return cols
    except Exception as e:
        log.error(f"read_header_line エラー: {e}")
        return None

    return cols


# ------------------------------------------------
# 開始行の列数取得
# ------------------------------------------------
def detect_column_count(path: str, sep: str, data_start_line: int):
    skip = max(data_start_line - 1, 0)
    sample = safe_read_csv( path,sep=sep,header=None,skiprows=skip,nrows=1,engine="python",on_bad_lines="skip",encoding='utf-8')
    return len(sample.columns)


# =========================================================
#  ★ 横連結コード
# =========================================================
class MergeSourceConfigFrame(ctk.CTkFrame):
    def __init__(self, master, index: int, app_ref):
        super().__init__(master)
        self.app = app_ref
        self.index = index

        # 連結有効/無効チェック
        self.var_enable = tk.BooleanVar(value=(index == 0))
        # csv区切り文字
        self.sep_display_to_actual = {
            "コンマ": ",",
            "タブ": "\t",
            "スペース": " "
        }

        # ------------------------------------------------
        # 1行目UI
        # ------------------------------------------------ 
        header_row = ctk.CTkFrame(self)
        header_row.pack(fill="x")
        ctk.CTkCheckBox(header_row, text=f"データセット {index+1}", variable=self.var_enable).pack(side="left")
        ctk.CTkButton(header_row, text="表示", width=60, command=self.show_files).pack(side="right", padx=3)
        self.lbl_status = ctk.CTkLabel(header_row, text="")
        self.lbl_status.pack(side="left",padx=10)

        # ------------------------------------------------
        # 2行目UI
        # ------------------------------------------------ 
        row2 = ctk.CTkFrame(self)
        row2.pack(fill="x", pady=(4,2))

        ctk.CTkLabel(row2, text="データセット名").pack(side="left")
        self.ent_prefix = ctk.CTkEntry(row2, width=120)
        self.ent_prefix.insert(0, f"D{index+1}_")
        self.ent_prefix.pack(side="left", padx=5)
        self.ent_prefix.bind("<KeyRelease>", lambda e: self._update_header_preview())
        self.ent_prefix.bind("<FocusOut>", lambda e: self._update_header_preview())

        ctk.CTkLabel(row2, text="ヘッダプレビュー").pack(side="left", padx=(10, 3))
        self.lbl_header_preview = ctk.CTkLabel(row2, text="(未読込)", anchor="w")
        self.lbl_header_preview.pack(side="left", padx=5, fill="x", expand=True)

        ctk.CTkLabel(row2, text="区切り").pack(side="left", padx=(10, 3))
        self.cmb_sep = ctk.CTkComboBox(row2, width=90, values=list(self.sep_display_to_actual.keys()))
        self.cmb_sep.set("コンマ")
        self.cmb_sep.pack(side="left", padx=2)

        ctk.CTkLabel(row2, text="開始行").pack(side="left", padx=(10, 3))
        self.ent_data_start = ctk.CTkEntry(row2,width=50)
        self.ent_data_start.insert(0, "1")
        self.ent_data_start.pack(side="left", padx=2)

        ctk.CTkLabel(row2, text="時間列番号").pack(side="left", padx=(10, 3))
        self.ent_time_col = ctk.CTkEntry(row2, width=50)
        self.ent_time_col.insert(0, "1")
        self.ent_time_col.pack(side="left", padx=2)

        # ------------------------------------------------
        # 3行目UI
        # ------------------------------------------------ 
        row3 = ctk.CTkFrame(self)
        row3.pack(fill="x", pady=2)

        ctk.CTkButton(row3, text="フォルダ選択", command=self.browse_folder).pack(side="left")
        self.ent_folder = ctk.CTkEntry(row3, width=260)
        self.ent_folder.pack(side="left", padx=5)
        ctk.CTkButton(row3, text="ファイル数確認", command=self.test_load).pack(side="left", padx=5)
        
        # ------------------------------------------------
        # 4行目UI
        # ------------------------------------------------ 
        row4 = ctk.CTkFrame(self)
        row4.pack(fill="x", pady=(2,6))

        ctk.CTkButton(row4, text="ヘッダファイル選択", command=self.browse_header).pack(side="left")
        self.ent_header_path = ctk.CTkEntry(row4, width=260)
        self.ent_header_path.pack(side="left", padx=5)
        ctk.CTkButton(row4, text="列数確認", command=self.load_header_from_entry).pack(side="left", padx=5)

        self.header_cols = None

    
    # ----------------- UI内の処理いろいろ -----------------
    def browse_folder(self):
        d = filedialog.askdirectory()
        if not d:
            return
        self.ent_folder.delete(0, "end")
        self.ent_folder.insert(0, d)

    def test_load(self):
        folder = self.ent_folder.get().strip()
        if not folder or not Path(folder).exists():
            self.lbl_status.configure(text="フォルダ無効")
            MSG_ERR("エラー", "フォルダパスが正しくありません")
            return
        files = list(Path(folder).glob("*.csv")) + list(Path(folder).glob("*.txt"))
        self.lbl_status.configure(text=f"{len(files)} ファイル")
        log.info(f"横結合セット{self.index+1}: {len(files)}ファイル")

    def browse_header(self):
        f = filedialog.askopenfilename(filetypes=[("CSV/TXT","*.csv *.txt")])
        if not f:
            return
        self.ent_header_path.delete(0, "end")
        self.ent_header_path.insert(0, f)
        self._load_header(f)

    def _update_header_preview(self):
        prefix = self.ent_prefix.get().strip() or f"D{self.index+1}_"
        if not prefix.endswith("_"):
            prefix += "_"
        
        cols = self.header_cols or []

        N = 3

        if cols:
            base = [c for c in cols if str(c).strip()][:]
            base_no_time = base[1:] if len(base) >= 2 else base
            sample = base_no_time[:N] if base_no_time else base[:N]
            preview_items = [prefix + str(c) for c in sample]
        else:
            preview_items = [f"{prefix}COL{i+1}" for i in range(N)]
        
        text = ", ".join(preview_items)
        if (cols and len(cols) > N+1) or (not cols):
            text += " ..."
        self.lbl_header_preview.configure(text=text)

    def _load_header(self, path: str):
        sep = self.sep_display_to_actual.get(self.cmb_sep.get(), ",")
        cols = read_header_line(path, sep)
        if cols:
            self.header_cols = cols
            self.lbl_status.configure(text=f"ヘッダ {len(cols)} 列")
            preview = ", ".join(cols[:2])
            self.lbl_header_preview.configure(text=preview)
            log.info(f"[横結合] セット{self.index+1} ヘッダ読み込み: {cols}")
        else:
            self.header_cols = None
            self.lbl_status.configure(text="ヘッダ読込失敗")
            self.lbl_header_preview.configure(text="(未読込)")
            log.warning(f"[横結合] セット{self.index+1} ヘッダ読み込み失敗")
        self._update_header_preview()
        

    def load_header_from_entry(self):
        path = self.ent_header_path.get().strip()
        if not path or not Path(path).exists():
            MSG_ERR("エラー", "ヘッダファイルパスが無効です")
            return
        self._load_header(path)

    def show_files(self):
        log.info("show_files pressed")
        params = self.get_params()
        if not params:
            MSG_ERR("エラー", f"DataSet {self.index+1} のパラメータが無効です")
            return

        folder = Path(params["folder"])
        sep = params["sep"]
        data_start = params["data_start"]

        files = list(folder.glob("*.csv")) + list(folder.glob("*.txt"))

        self.app.update_hmerge_file_view(
            idx=self.index+1,
            files=files,
            sep=sep,
            data_start=data_start
        )

    
    def set_params(self, params: dict):
        """config.json から読み込んだ設定をUIへ反映"""
        if not params:
            return

        self.var_enable.set(params.get("enable", False))

        folder = params.get("folder", "")
        self.ent_folder.delete(0, "end")
        self.ent_folder.insert(0, folder)

        # sep は表示名に戻す
        sep_rev = {v:k for k,v in self.sep_display_to_actual.items()}
        disp = sep_rev.get(params.get("sep", ","), "コンマ")
        self.cmb_sep.set(disp)

        self.ent_data_start.delete(0, "end")
        self.ent_data_start.insert(0, str(params.get("data_start", 1)))

        header_path = params.get("header_path", "")
        self.ent_header_path.delete(0, "end")
        self.ent_header_path.insert(0, header_path)

        # header_cols はそのまま復元（内部変数のみ）
        self.header_cols = params.get("header_cols", None)

        if hasattr(self, "ent_prefix"):
            self.ent_prefix.delete(0, "end")
            self. ent_prefix.insert(0, params.get("prefix", f"D{self.index+1}_"))

        if hasattr(self, "ent_time_col"):
            self.ent_time_col.delete(0, "end")
            self.ent_time_col.insert(0, str(params.get("time_col_index", 1)))

        cols = params.get("header_cols")
        if cols:
            preview = ", ".join(cols[:4])
            if len(cols) > 4:
                preview += " ..."
            self.lbl_header_preview.configure(text=preview)
        else:
            self.lbl_header_preview.configure(text="(未読込)")

        if hasattr(self, "lbl_header_preview"):
            self._update_header_preview()


    # ----------------- CSVApp 側から参照するための getter -----------------
    def get_params(self):
        """有効かつフォルダ指定されていれば設定を dict で返す。無効なら None"""
        if not self.var_enable.get():
            return None

        folder = self.ent_folder.get().strip()
        if not folder or not Path(folder).exists():
            return None
        
        sep = self.sep_display_to_actual.get(self.cmb_sep.get(), ",")

        try:
            data_start = int(self.ent_data_start.get())
        except:
            data_start = 1
        data_start = max(data_start, 1)

        try:
            time_col_index_1based = int(self.ent_time_col.get())
        except:
            time_col_index_1based = 1
        time_col_index_1based = max(time_col_index_1based, 1)

        prefix = self.ent_prefix.get().strip() or f"D{self.index+1}_"
        if not prefix.endswith("_"):
            prefix += "_"

        return {
            "folder": folder,
            "sep": sep,
            "data_start": data_start,
            "header_path": self.ent_header_path.get().strip(),
            "header_cols": self.header_cols,
            "prefix": prefix,
            "time_col_index": time_col_index_1based,
        }
    

# =========================================================
#  ★ メインアプリ
# =========================================================
class CSVApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.config = load_config()
        self.bit_map_path = self.config.get("map")
        self.last_sep = self.config.get("separator", "comma")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("CSV解析ツール")
        self.geometry("1400x830")

        self.df_current=None
        self.csv_files=[]
        self.df_bit=None
        self.df_map=None
        self.bit_map_path=None
        self.merge_header_cols=None
        self.merge_header_path=self.config.get("merge_header")

        # 結合用ヘッダの読み込み
        if self.merge_header_path and Path(self.merge_header_path).exists():
            try:
                df_header = safe_read_csv(self.merge_header_path, header=None)
                self.merge_header_cols = df_header.iloc[0].tolist()
                log.info(f"* 起動時ヘッダ復元成功 → {self.merge_header_cols}")
            except Exception as e:
                log.error(f"起動時ヘッダ読み取り失敗: {e}")

        # ------- タブ -------
        self.tab = ctk.CTkTabview(self)
        self.tab.pack(fill="both",expand=True,padx=10,pady=10)

        self.tab.add("グラフ作成")
        self.tab.add("CSV結合")
        self.tab.add("CSV横結合")
        self.tab.add("ビット変換")

        self._setup_graph_tab()
        self._setup_merge_tab()
        self._setup_merge_horizontal_tab()
        self._setup_bit_tab()

        saved = self.config.get("hmerge_sources", [])
        for src, params in zip(self.hmerge_sources, saved):
            src.set_params(params)


    # =========================================================
    # ■ グラフ作成タブ
    # =========================================================
    def _setup_graph_tab(self):

        frame=self.tab.tab("グラフ作成")

        left=ctk.CTkFrame(frame,width=350)
        left.pack(side="left",fill="y",padx=10,pady=10)
        right=ctk.CTkFrame(frame)
        right.pack(side="left",fill="both",expand=True,padx=10,pady=10)


        # --- CSV選択 + パス表示
        row=ctk.CTkFrame(left)
        row.pack(fill="x",pady=5)
        ctk.CTkButton(row,text="CSV選択",command=self.select_csv_graph).pack(side="left",padx=4)
        self.lbl_graph_path=ctk.CTkLabel(row,text="未選択")
        self.lbl_graph_path.pack(side="left",fill="x",expand=True)


        # X・Y設定
        ctk.CTkLabel(left,text="X軸").pack()
        self.cmb_x=ctk.CTkComboBox(left,values=[])
        self.cmb_x.pack(fill="x")

        ctk.CTkLabel(left,text="Y軸（複数選択）").pack(pady=(8,2))
        self.frame_y=ctk.CTkFrame(left)
        self.frame_y.pack(fill="x")

        #ラベル
        for t,n in [("タイトル","ttl"),("X軸ラベル","xl"),("Y軸ラベル","yl")]:
            ctk.CTkLabel(left,text=t).pack()
            setattr(self,f"ent_{n}",ctk.CTkEntry(left))
            getattr(self,f"ent_{n}").pack(fill="x",padx=5)


        #--- スライダー ---
        ctk.CTkLabel(left,text="X範囲").pack(pady=(10,2))
        self.s_min=ctk.CTkSlider(left,command=self.update_slider)
        self.s_max=ctk.CTkSlider(left,command=self.update_slider)
        self.s_min.pack(fill="x");self.s_max.pack(fill="x",pady=3)
        self.lbl_range=ctk.CTkLabel(left,text="-")
        self.lbl_range.pack(pady=3)

        ctk.CTkButton(left,text="描画",command=self.draw_graph).pack(pady=6)


        #--- プロット領域 ---
        self.fig,self.ax=plt.subplots(figsize=(7,5))
        self.canvas=FigureCanvasTkAgg(self.fig,master=right)
        self.canvas.get_tk_widget().pack(fill="both",expand=True)



    # ▼ CSV選択
    def select_csv_graph(self):
        f=filedialog.askopenfilename(filetypes=[("CSV/TXT","*.csv *.txt")])
        if not f:return
        self.lbl_graph_path.configure(text=f)

        df=read_data_auto(f)
        self.df_current=df
        cols=list(df.columns)
        self.cmb_x.configure(values=cols)
        self.cmb_x.set("TIME" if "TIME" in cols else cols[0])

        for w in self.frame_y.winfo_children():w.destroy()
        self.y_vars=[]
        for c in cols:
            if c!=self.cmb_x.get():
                v=tk.BooleanVar(value=True)
                ctk.CTkCheckBox(self.frame_y,text=c,variable=v).pack(anchor="w")
                self.y_vars.append((c,v))

        x=self.cmb_x.get()
        if "TIME" in x or pd.api.types.is_datetime64_any_dtype(df[x]):
            df[x]=pd.to_datetime(df[x],errors="coerce")
            base=df[x].min()
            sec=(df[x]-base).dt.total_seconds()
        else:
            sec=df[x]

        mn,mx=float(sec.min()),float(sec.max())
        self.s_min.configure(from_=mn,to=mx,value=mn)
        self.s_max.configure(from_=mn,to=mx,value=mx)
        self.update_slider()


    def update_slider(self,*_):
        mn,mx=self.s_min.get(),self.s_max.get()
        if mn>mx:mn,mx=mx,mn;self.s_min.set(mn);self.s_max.set(mx)
        self.lbl_range.configure(text=f"{mn:.1f}〜{mx:.1f}")


    def draw_graph(self):
        if self.df_current is None:return

        df=self.df_current.copy()
        xcol=self.cmb_x.get()
        df[xcol]=pd.to_datetime(df[xcol],errors="coerce")
        base=df[xcol].min()
        sec=(df[xcol]-base).dt.total_seconds()

        mn,mx=self.s_min.get(),self.s_max.get()
        m=(sec>=mn)&(sec<=mx)

        ys=[c for c,v in self.y_vars if v.get()]
        if not ys:return

        self.ax.clear()
        for y in ys:self.ax.plot(df[xcol][m],df[y][m],label=y)
        self.ax.set_title(self.ent_ttl.get() or ", ".join(ys))
        self.ax.set_xlabel(self.ent_xl.get() or xcol)
        self.ax.set_ylabel(self.ent_yl.get())
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()



    # =========================================================
    # ■ CSV結合タブ
    # =========================================================
    def _setup_merge_tab(self):

        frame=self.tab.tab("CSV結合")

        top=ctk.CTkFrame(frame)
        top.pack(fill="x",padx=10,pady=10)

        #------------------------------------------------------
        # ▼ フォルダ選択
        #------------------------------------------------------
        row_folder = ctk.CTkFrame(frame)
        row_folder.pack(fill="x", padx=10, pady=(5,5))
        #[フォルダ選択ボタン]
        ctk.CTkButton(top,text="フォルダ選択",command=self.select_merge_folder).pack(side="left")
        #[フォルダ入力欄]
        self.ent_folder_path = ctk.CTkEntry(row_folder, width=450)
        self.ent_folder_path.pack(side="left",padx=8)
        if "merge_folder" in self.config:
            self.ent_folder_path.insert(0, self.config["merge_folder"])
        #[フォルダ読み込みボタン]
        ctk.CTkButton(row_folder, text="読込", command=self.load_folder_from_entry).pack(side="left", padx=8)

        #------------------------------------------------------
        # ▼ 区切り選択
        #------------------------------------------------------
        ctk.CTkLabel(top,text="区切り").pack(side="left", padx=(20,5))
        self.sep_display_to_actual = {
            "コンマ":   ("comma", ","),
            "タブ":     ("tab", "\t"),
            "スペース": ("space", " ")
        }

        self.cmb_sep = ctk.CTkComboBox(
            top,width=120,
            values=list(self.sep_display_to_actual.keys())
        )
        reverse_lookup = {"comma":"コンマ","tab":"タブ","space":"スペース"}
        self.cmb_sep.set(reverse_lookup.get(self.last_sep, "コンマ"))
        self.cmb_sep.pack(side="left")

        #------------------------------------------------------
        # ▼ データ開始行
        #------------------------------------------------------
        ctk.CTkLabel(top,text="データ開始行").pack(side="left", padx=(20,5))
        self.ent_data_start = ctk.CTkEntry(top, width=60)
        self.ent_data_start.insert(0, str(self.config.get("data_start_line", 1)))
        self.ent_data_start.pack(side="left")

        #------------------------------------------------------
        # ▼ ヘッダファイル選択
        #------------------------------------------------------
        header_row = ctk.CTkFrame(frame)
        header_row.pack(fill="x", padx=10, pady=(5,10))
        #[ファイル選択ボタン]
        ctk.CTkButton(header_row, text="ヘッダファイル選択", command=self.select_merge_header).pack(side="left")
        #[ファイルパス入力欄]
        self.ent_header_path = ctk.CTkEntry(header_row, width=450)
        self.ent_header_path.pack(side="left", padx=8)
        if "merge_header" in self.config:
            self.ent_header_path.insert(0, self.config["merge_header"])
        #[ファイル読み込みボタン]
        ctk.CTkButton(header_row, text="読込", command=self.load_header_from_entry).pack(side="left", padx=8)

        #------------------------------------------------------
        # ▼ 保存先
        #------------------------------------------------------
        row=ctk.CTkFrame(frame);row.pack(fill="x",padx=10,pady=(2,10))
        ctk.CTkLabel(row,text="保存名").pack(side="left")
        self.ent_merge_name=ctk.CTkEntry(row,width=150);self.ent_merge_name.insert(0,"結合データ")
        self.ent_merge_name.pack(side="left",padx=5)

        ctk.CTkLabel(row,text="保存先").pack(side="left")
        self.ent_merge_out=ctk.CTkEntry(row,width=350);self.ent_merge_out.pack(side="left",padx=5)
        ctk.CTkButton(row,text="選択",command=self.select_merge_save).pack(side="left",padx=5)

        ctk.CTkButton(frame,text="結合実行",command=self.run_merge).pack(pady=5)

        # ▼ ファイル一覧＋開始/終了表示
        main=ctk.CTkFrame(frame);main.pack(fill="both",expand=True,padx=10,pady=10)

        self.txt_merge=ctk.CTkScrollableFrame(main, width=260, height=500)
        self.txt_merge.pack(side="left",fill="y",padx=(0,8))

        right=ctk.CTkFrame(main);right.pack(side="left",fill="both",expand=True)
        head=ctk.CTkFrame(right);head.pack(fill="x")
        ctk.CTkLabel(head,text="ファイル",width=240).pack(side="left")
        ctk.CTkLabel(head,text="開始",width=200).pack(side="left")
        ctk.CTkLabel(head,text="終了",width=200).pack(side="left")

        canvas_frame=ctk.CTkFrame(right);canvas_frame.pack(fill="both",expand=True)
        self.canvas_merge=tk.Canvas(canvas_frame,highlightthickness=0)
        sb=tk.Scrollbar(canvas_frame,command=self.canvas_merge.yview)
        self.canvas_merge.configure(yscrollcommand=sb.set)
        sb.pack(side="right",fill="y")
        self.canvas_merge.pack(side="left",fill="both",expand=True)

        self.inner_merge=ctk.CTkFrame(self.canvas_merge)
        self.canvas_merge.create_window((0,0),window=self.inner_merge,anchor="nw")
        self.inner_merge.bind("<Configure>",lambda e:self.canvas_merge.configure(scrollregion=self.canvas_merge.bbox("all")))
        self.canvas_merge.bind("<Configure>",lambda e:self.canvas_merge.itemconfig("all",width=e.width))


    def _get_current_sep_char(self):
        disp = self.cmb_sep.get()
        key, sep = self.sep_display_to_actual.get(disp, ("comma", ","))
        return sep

    def _get_data_start_line(self):
        try:
            n = int(self.ent_data_start.get())
        except:
            n = 1
        return max(n, 1)

    def select_merge_header(self):
        f = filedialog.askopenfilename(filetypes=[("CSV/TXT","*.csv *.txt")])
        if not f:
            return
        self.merge_header_path = f
        self.ent_header_path.delete(0, "end")
        self.ent_header_path.insert(0,f)

        sep = self._get_current_sep_char()
        cols = read_header_line(f, sep)
        if cols:
            self.merge_header_cols = cols
            log.info(f"ヘッダ読み込み: {cols}")
            # コンフィグに保存
            self.config["merge_header"] = f
            save_config(self.config)
        else:
            log.warning("ヘッダファイルの読み込みに失敗しました")
            MSG_ERR("ヘッダエラー","ヘッダファイルを読み込めませんでした")

    
    def load_header_from_entry(self):
        path = self.ent_header_path.get().strip()
        if not path or not Path(path).exists():
            return MSG_ERR("エラー", "ヘッダファイルパスが無効です")

        sep = self._get_current_sep_char()
        cols = read_header_line(path, sep)

        if cols:
            self.merge_header_cols = cols
            self.config["merge_header"]=path
            save_config(self.config)
            log.info(f"✔ ヘッダ読み込み(手入力) → {cols}")
            MSG_OK("完了","ヘッダ読み込み成功")
        else:
            MSG_ERR("読み込み失敗","ヘッダを読み込めませんでした")
    

    def load_folder_from_entry(self):
        folder = self.ent_folder_path.get().strip()
        if not folder or not Path(folder).exists():
            return MSG_ERR("エラー","パスが正しくありません")
        
        self.config["merge_folder"] = folder
        save_config(self.config)

        self._load_merge_files(folder)
    

    def select_merge_folder(self):
        d = filedialog.askdirectory()
        if not d:return

        self.ent_folder_path.delete(0,"end")
        self.ent_folder_path.insert(0,d)

        self.config["merge_folder"] = d
        save_config(self.config)

        self._load_merge_files(d)

    def _load_merge_files(self, d):
        self.ent_merge_out.delete(0,"end")
        self.ent_merge_out.insert(0,d)
        self.csv_files=sorted(list(Path(d).glob("*.csv")) + list(Path(d).glob("*.txt")))

        for w in self.txt_merge.winfo_children(): 
            w.destroy()

        for w in self.inner_merge.winfo_children():
            w.destroy()

        sep_char = self._get_current_sep_char()
        data_start = self._get_data_start_line()

        for i,f in enumerate(self.csv_files):
            self._add_merge_text(f"{i+1:3d}  {f.name}")

            st,ed=get_start_end(f, sep_char, data_start)
            row=ctk.CTkFrame(self.inner_merge);row.pack(fill="x",padx=3,pady=1)
            ctk.CTkLabel(row,text=f.name,width=240).pack(side="left")
            s=ctk.CTkEntry(row,width=200);s.insert(0,st);s.configure(state="readonly");s.pack(side="left")
            e=ctk.CTkEntry(row,width=200);e.insert(0,ed);e.configure(state="readonly");e.pack(side="left")


    def _add_merge_text(self, text):
        ctk.CTkLabel(self.txt_merge, text=text, anchor="w").pack(fill="x")

    def select_merge_save(self):
        d=filedialog.askdirectory()
        if d:self.ent_merge_out.delete(0,"end");self.ent_merge_out.insert(0,d)


    # ▼▼▼ CSV結合 ▼▼▼
    def run_merge(self):
        try:
            if not self.csv_files:
                log.warning("結合対象無し")
                return
            
            disp = self.cmb_sep.get()
            key, sep = self.sep_display_to_actual[disp]
            data_start = self._get_data_start_line()            

            # configに設定保存
            self.config["separator"] = key
            self.config["data_start_line"] = data_start
            save_config(self.config)

            out=Path(self.ent_merge_out.get())/f"{self.ent_merge_name.get()}.csv"
            log.info(f"▶︎ CSV結合開始 sep={repr(sep)} 出力先={out} 開始行={data_start}")

            skip = max(data_start -1, 0)

            dfs=[]

            first_cols = None
            if self.csv_files:
                try:
                    first_cols = detect_column_count(str(self.csv_files[0]), sep, data_start)
                    log.info(f"[縦結合]代表列数 = {first_cols}")
                except Exception as e:
                    log.warning(f"[縦結合]列数見地に失敗: {e}")

            for f in self.csv_files:
                df = safe_read_csv(f,sep=sep,encoding="utf-8",header=None,skiprows=skip,engine="python",on_bad_lines="skip")
                log.info(f"[縦結合] {f.name}: 読み込み {df.shape[0]} 行 / {df.shape[1]} 列")
                if first_cols is not None and df.shape[1] != first_cols:
                    log.warning(f"[縦結合] 列数不一致: {f.name} → {df.shape[1]}列 (期待 {first_cols})")
                dfs.append(df)
            
            if not dfs:
                MSG_ERR("エラー","有効なデータがありません")
                return

            merged = pd.concat(dfs,ignore_index=True)

            # ヘッダファイルが指定されていたら列名を適用
            if self.merge_header_cols:
                if len(self.merge_header_cols) == merged.shape[1]:
                    merged.columns = self.merge_header_cols
                else :
                    log.warning(f"ヘッダ列数({len(self.merge_header_cols)})と列数が一致しません。ヘッダ適用をスキップします。")

            merged.to_csv(out,index=False,encoding="utf-8-sig")

            log.info(f"✔️ 結合完了 → {out}")
            MSG_OK("完了", str(out))
            
        except Exception as e:
            log.exception("⚠️ run_merge()中にエラー発生")
            MSG_ERR("ERROR", str(e))


    # =========================================================
    # ■ CSV横結合タブ
    # =========================================================
    def _setup_merge_horizontal_tab(self):
        frame = self.tab.tab("CSV横結合")

        # --- 上部：キー列 ＋ 保存名/保存先 ---
        top = ctk.CTkFrame(frame)
        top.pack(fill="x", padx=10, pady=10)

        # --- 実行ボタン ---
        ctk.CTkButton(top, text="横結合実行", command=self.run_merge_horizontal).pack(side="left")

        # 保存名＆保存先（縦結合と同じ構成）
        row_save = ctk.CTkFrame(frame)
        row_save.pack(fill="x", padx=10, pady=(0,10))

        ctk.CTkLabel(row_save, text="保存名").pack(side="left")
        self.ent_hmerge_name = ctk.CTkEntry(row_save, width=150)
        self.ent_hmerge_name.insert(0, "横結合データ")
        self.ent_hmerge_name.pack(side="left", padx=5)

        ctk.CTkLabel(row_save, text="保存先").pack(side="left")
        self.ent_hmerge_out = ctk.CTkEntry(row_save, width=350)
        self.ent_hmerge_out.pack(side="left", padx=5)
        ctk.CTkButton(row_save, text="選択", command=self.select_hmerge_save).pack(side="left", padx=5)

        # --- 左エリア：10セットのデータセット設定ブロック ---
        sources_frame = ctk.CTkScrollableFrame(frame, width=700)
        sources_frame.pack(side="left", fill="y", expand=False, padx=10, pady=10)

        self.hmerge_sources = []
        for i in range(10):
            block = MergeSourceConfigFrame(sources_frame, index=i, app_ref=self)
            block.pack(fill="x", pady=6)
            self.hmerge_sources.append(block)

        # --- 右エリア：ファイル表示パネル（横スクロール対応） ---
        right_panel = ctk.CTkFrame(frame)
        right_panel.pack(side="right", fill="both", expand=True, padx=5, pady=10)

        ctk.CTkLabel(right_panel, text="📄 表示結果", font=("Meiryo",14,"bold")).pack()

        self.hmerge_file_view = ctk.CTkScrollableFrame(
            right_panel,
            width=450,
            height=700,
            #orientation=""     # ★ 横スクロール有効化
        )
        self.hmerge_file_view.pack(fill="both", expand=True)


    def set_params(self, params: dict):
        """config.json から読み込んだ設定をUIへ反映"""
        if not params:
            return

        self.var_enable.set(params.get("enable", False))

        folder = params.get("folder", "")
        self.ent_folder.delete(0, "end")
        self.ent_folder.insert(0, folder)

        # sep は表示名に戻す
        sep_rev = {v:k for k,v in self.sep_display_to_actual.items()}
        disp = sep_rev.get(params.get("sep", ","), "コンマ")
        self.cmb_sep.set(disp)

        self.ent_data_start.delete(0, "end")
        self.ent_data_start.insert(0, str(params.get("data_start", 1)))

        header_path = params.get("header_path", "")
        self.ent_header_path.delete(0, "end")
        self.ent_header_path.insert(0, header_path)

        # header_cols はそのまま復元（内部変数のみ）
        self.header_cols = params.get("header_cols", None)


    def select_hmerge_save(self):
        d = filedialog.askdirectory()
        if d:
            self.ent_hmerge_out.delete(0, "end")
            self.ent_hmerge_out.insert(0, d)


    def update_hmerge_file_view(self, idx, files, sep, data_start):
        print("update呼び出し", idx, len(files))
        for w in self.hmerge_file_view.winfo_children():
            w.destroy()

        ctk.CTkLabel(self.hmerge_file_view, text=f"▼ DataSet {idx}",
                    font=("Meiryo",13,"bold")).pack(anchor="w", pady=(0,3))

        skip = max(data_start-1, 0)

        for f in files:
            st, ed = get_start_end(f, sep, data_start)
            row = ctk.CTkFrame(self.hmerge_file_view)
            row.pack(fill="x", pady=1)

            # 横スクロール前提 → 列拡張可能
            ctk.CTkLabel(row,text=f.name,width=200,anchor="w").pack(side="left")
            ctk.CTkLabel(row,text=st,width=100).pack(side="left",padx=10)
            ctk.CTkLabel(row,text=ed,width=100).pack(side="left",padx=10)


    
    # ▼▼▼ CSV横結合 ▼▼▼
    def run_merge_horizontal(self):
        try:
            log.info("▶︎ CSV横結合開始（共通期間 + __k__キー）")

            # =====================================================
            # ★ 設定（まずはここを固定でOK。後でUI化できます）
            # =====================================================
            PERIOD_S_MANUAL = 1.0   # ←まずは 1.0 固定推奨。将来UIで 0.1/0.2/0.5/5/10 等選択へ
            USE_AUTO_PERIOD = False # Trueにすると推定値を使う（推奨はFalse）
            ROUNDING = "round"      # "round" | "floor" | "ceil"
            TOLERANCE_S = None      # 例: 0.25 など（±0.25秒超を除外）。まずNoneでOK
            JOIN_MODE = "inner"     # "inner"推奨：欠損が出る行は落とす / "outer"：欠損許容
            # =====================================================

            # ------------------------------------------------------
            # 1回目パス：開始/終了を集める（共通期間を決める）
            # ------------------------------------------------------
            datasets = []     # 後で再利用するので、読み込んだdf等を保持
            start_times = []
            end_times = []

            for idx, src in enumerate(self.hmerge_sources, start=1):
                params = src.get_params()
                if not params:
                    continue

                folder = Path(params["folder"])
                sep = params["sep"]
                data_start = params["data_start"]
                header_cols = params["header_cols"]

                files = sorted(list(folder.glob("*.csv")) + list(folder.glob("*.txt")))
                if not files:
                    log.warning(f"[横結合] データセット{idx}: CSV/TXTファイルなし")
                    continue

                skip = max(data_start - 1, 0)
                df_list = []
                for f in files:
                    df = safe_read_csv(f, sep=sep, encoding="utf-8", header=None, skiprows=skip)
                    df_list.append(df)

                if not df_list:
                    continue

                df = pd.concat(df_list, ignore_index=True)

                # ヘッダ適用
                if header_cols and len(header_cols) == df.shape[1]:
                    df.columns = header_cols
                elif header_cols:
                    log.warning(f"[横結合] データセット{idx}: ヘッダ列数不一致のため適用スキップ")

                if key_col_index >= len(df.columns):
                    MSG_ERR("エラー", f"データセット{idx}: 時間列 {key_col_index+1} 列目が存在しません")
                    return

                time_col_index = int(params.get("time_col_index", 1)) - 1
                if time_col_index < 0 or time_col_index >= len(df.columns):
                    MSG_ERR("エラー", f"データセット{idx}: 時間列 {time_col_index+1} 列目が存在しません")
                    return

                df.columns = [str(c) for c in df.columns]

                time_col = df.columns[key_col_index]
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

                st = df[time_col].min()
                ed = df[time_col].max()
                if pd.notna(st) and pd.notna(ed):
                    start_times.append(st)
                    end_times.append(ed)

                datasets.append({
                    "idx": idx,
                    "df": df,
                    "time_col": time_col,
                    "prefix": params.get("prefix", f"D{idx}_"),
                })

            if not datasets:
                MSG_ERR("エラー", "有効なデータセットがありません")
                return
            if not start_times or not end_times:
                MSG_ERR("エラー", "開始/終了時刻を取得できませんでした（時間列の解析に失敗している可能性）")
                return

            # 共通期間（あなたの要件）
            final_start = max(start_times)
            final_end = min(end_times)

            if final_start >= final_end:
                MSG_ERR("エラー", f"共通期間が成立しません: start={final_start} end={final_end}")
                return

            log.info(f"▼ 共通期間: {final_start} ~ {final_end}")

            # ------------------------------------------------------
            # 2回目パス：共通期間に切り出し → __k__生成 → プレフィックス付与
            # ------------------------------------------------------
            dfs_ready = []
            period_s_list = []

            for ds in datasets:
                idx = ds["idx"]
                df = ds["df"]
                time_col = ds["time_col"]

                # 共通期間に切り出し
                d = df[(df[time_col] >= final_start) & (df[time_col] <= final_end)].copy()
                if d.empty:
                    log.warning(f"[横結合] データセット{idx}: 共通期間にデータがありません")
                    continue

                # 周期決定（まずは固定推奨）
                if USE_AUTO_PERIOD:
                    period_s = estimate_period_seconds(d[time_col])
                else:
                    period_s = float(PERIOD_S_MANUAL)

                period_s_list.append(period_s)

                # ★ 共通基準時刻は final_start に統一（重要）
                d = add_sample_index(
                    d,
                    time_col=time_col,
                    t_ref=final_start,
                    period_s=period_s,
                    rounding=ROUNDING,
                    tolerance_s=TOLERANCE_S
                )

                if d.empty:
                    log.warning(f"[横結合] データセット{idx}: __k__生成後に有効データがありません")
                    continue

                # 衝突回避：時間列と__k__以外にプレフィックス
                prefix = ds.get("prefix") or f"D{idx}_"
                if not prefix.endswith("_"):
                    prefix += "_"

                rename_map = {c: prefix + c for c in d.columns if c not in [time_col, "__k__"]}
                d = d.rename(columns=rename_map)

                # 並び：__k__ → TIME → その他
                cols = ["__k__", time_col] + [c for c in d.columns if c not in ["__k__", time_col]]
                d = d[cols]

                dfs_ready.append(d)

                log.info(f"[横結合] データセット{idx}: period={period_s}s, rows={len(d)}")

            if not dfs_ready:
                MSG_ERR("エラー", "共通期間に有効なデータがありません")
                return

            # ------------------------------------------------------
            # 結合：キーは __k__（時刻の完全一致を要求しない）
            # ------------------------------------------------------
            merged = dfs_ready[0]
            for d in dfs_ready[1:]:
                merged = pd.merge(merged, d, on="__k__", how=JOIN_MODE)

            # 揃った代表時刻列を生成（出力を扱いやすく）
            # ※周期が混在する場合は要注意だが、同周期前提ならOK
            base_period = float(np.median(period_s_list)) if period_s_list else float(PERIOD_S_MANUAL)
            merged["TIME_ALIGNED"] = final_start + pd.to_timedelta(merged["__k__"] * base_period, unit="s")

            # ------------------------------------------------------
            # 最終出力を整形：TIME列１本化、不要列の削除
            # ------------------------------------------------------
            base_time_cols = sorted({ds["time_col"] for ds in datasets})
            cols_to_drop = []
            for c in merged.columns:
                for base in base_time_cols:
                    if c == base or c.startswith(base + "_"):
                        cols_to_drop.append(c)
                        break
            
            cols_to_drop = [c for c in cols_to_drop if c != "TIME_ALIGNED"]

            if cols_to_drop:
                merged = merged.drop(columns=cols_to_drop)

            if "__k__" in merged.columns:
                merged = merged.drop(columns=["__k__"])

            if "TIME_ALIGNED" in merged.columns:
                cols = ["TIME_ALIGNED"] + [c for c in merged.columns if c != "TIME_ALIGNED"]
                merged = merged[cols]


            # ------------------------------------------------------
            # 保存
            # ------------------------------------------------------
            out_dir = Path(self.ent_hmerge_out.get().strip() or ".")
            out_dir.mkdir(parents=True, exist_ok=True)
            name = self.ent_hmerge_name.get().strip() or "横結合データ"
            out_path = out_dir / f"{name}.csv"

            merged.to_csv(out_path, index=False, encoding="utf-8-sig")
            log.info(f"✔️ 横結合完了 → {out_path}")
            MSG_OK("完了", str(out_path))

        except Exception as e:
            log.exception("⚠️ run_merge_horizontal()中にエラー発生")
            MSG_ERR("ERROR", str(e))

        # 横結合設定を保存（あなたの元コードのまま）
        cfg_sources = []
        for src in self.hmerge_sources:
            p = src.get_params()
            if p:
                p["enable"] = True
            else:
                p = {"enable": False}
            cfg_sources.append(p)

        self.config["hmerge_sources"] = cfg_sources
        save_config(self.config)



    # =========================================================
    # ■ ビット変換タブ
    # =========================================================
    def _setup_bit_tab(self):

        frame=self.tab.tab("ビット変換")

        row1=ctk.CTkFrame(frame);row1.pack(fill="x",padx=10,pady=5)
        ctk.CTkButton(row1,text="CSV選択",command=self.select_bit_csv).pack(side="left")
        self.lbl_bit_csv=ctk.CTkLabel(row1,text="未選択");self.lbl_bit_csv.pack(side="left",padx=6)

        row2=ctk.CTkFrame(frame);row2.pack(fill="x",padx=10,pady=5)
        ctk.CTkButton(row2,text="変換MAP",command=self.select_bit_map).pack(side="left")
        self.lbl_bit_map=ctk.CTkLabel(row2,text=self.bit_map_path or "未選択")
        self.lbl_bit_map.pack(side="left",padx=6)

        row3=ctk.CTkFrame(frame);row3.pack(fill="x",padx=10,pady=5)
        ctk.CTkLabel(row3,text="保存先").pack(side="left")
        self.ent_bit_out=ctk.CTkEntry(row3,width=400);self.ent_bit_out.pack(side="left",padx=8)
        ctk.CTkButton(row3,text="選択",command=self.select_bit_out).pack(side="left")

        ctk.CTkButton(frame,text="変換実行",command=self.run_bit).pack(pady=10)
        self.lbl_bit_state=ctk.CTkLabel(frame,text="待機中");self.lbl_bit_state.pack()



    def select_bit_csv(self):
        f=filedialog.askopenfilename(filetypes=[("CSV/TXT","*.csv *.txt")])
        if not f:return
        self.lbl_bit_csv.configure(text=f)
        self.df_bit=read_data_auto(f)
        self.ent_bit_out.delete(0,"end");self.ent_bit_out.insert(0,str(Path(f).parent))


    def select_bit_map(self):
        f=filedialog.askopenfilename(filetypes=[("CSV","*.csv")])
        if not f:return
        self.lbl_bit_map.configure(text=f)
        self.bit_map_path=f
        self.df_map=read_data_auto(f)
        self.config["map"]=f
        save_config(self.config)


    def select_bit_out(self):
        d=filedialog.askdirectory()
        if d:self.ent_bit_out.delete(0,"end");self.ent_bit_out.insert(0,d)


    def run_bit(self):
        try:
            if self.df_bit is None or self.df_map is None:
                return MSG_ERR("エラー","CSV or MAP未選択")

            df=self.df_bit.copy()
            for bit_index,r in self.df_map.iterrows():
                o=str(r[0]).strip()
                n=str(r[1]).strip()
                if o not in df.columns: continue
                df[n]=df[o].apply(lambda x:(int(x)>>bit_index)&1)

            out=Path(self.ent_bit_out.get())/f"{Path(self.lbl_bit_csv.cget('text')).stem}_BIT.csv"
            df.to_csv(out,index=False,encoding="utf-8-sig")

            log.info(f"✔️ ビット変換完了 → {out}")
            self.lbl_bit_state.configure(text="完了")
            MSG_OK("OK",str(out))

        except Exception as e:
            log.exception("⚠️ run_bit()中にエラー")
            self.lbl_bit_state.configure(text="エラー")
            MSG_ERR("ERROR",str(e))



# -----------------------------
# 実行
# -----------------------------
if __name__=="__main__":
    app=CSVApp()
    app.mainloop()
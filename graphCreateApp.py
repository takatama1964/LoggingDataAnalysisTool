import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


CONFIG_FILE = "bit_config.json"   # ビット変換MAP保存


# ------------------------------------------------
# CSV/TXT読込（UTF-16 / UTF-8 自動判定）
# ------------------------------------------------
def read_data_auto(path):
    enc = ["utf-8", "utf-16", "utf-16-le", "utf-16-be"]
    for e in enc:
        try:
            return pd.read_csv(path, encoding=e)
        except:
            pass
    return pd.read_csv(path, encoding="utf-8", errors="ignore") # 最終


# ------------------------------------------------
# 開始/終了取得（CSV結合で使用）UTF-16対応
# ------------------------------------------------
def get_start_end(path):
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

    start = next((l.split(",")[0].strip() for l in lines if l.strip() and "TIME" not in l),"??")
    end   = next((l.split(",")[0].strip() for l in reversed(lines) if l.strip()),"??")
    return start,end



# =========================================================
#  ★ メインアプリ
# =========================================================
class CSVApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.title("CSV解析ツール")
        self.geometry("1400x830")

        self.df_current=None
        self.csv_files=[]
        self.df_bit=None
        self.df_map=None
        self.bit_map_path=None

        # JSONロード
        if Path(CONFIG_FILE).exists():
            try:self.bit_map_path=json.load(open(CONFIG_FILE)).get("map")
            except:pass

        # ------- タブ -------
        self.tab = ctk.CTkTabview(self)
        self.tab.pack(fill="both",expand=True,padx=10,pady=10)

        self.tab.add("グラフ作成")
        self.tab.add("CSV結合")
        self.tab.add("ビット変換")

        self._setup_graph_tab()
        self._setup_merge_tab()
        self._setup_bit_tab()



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
    # ■ CSV結合 + txt対応 + 開始/終了表示
    # =========================================================
    def _setup_merge_tab(self):

        frame=self.tab.tab("CSV結合")

        top=ctk.CTkFrame(frame)
        top.pack(fill="x",padx=10,pady=10)
        ctk.CTkButton(top,text="フォルダ選択",command=self.select_merge_folder).pack(side="left")
        self.lbl_merge_folder=ctk.CTkLabel(top,text="未選択");self.lbl_merge_folder.pack(side="left",padx=10)

        # 保存指定
        row=ctk.CTkFrame(frame);row.pack(fill="x",padx=10,pady=(2,10))
        ctk.CTkLabel(row,text="保存名").pack(side="left")
        self.ent_merge_name=ctk.CTkEntry(row,width=150);self.ent_merge_name.insert(0,"結合データ")
        self.ent_merge_name.pack(side="left",padx=5)

        ctk.CTkLabel(row,text="保存先").pack(side="left")
        self.ent_merge_out=ctk.CTkEntry(row,width=350);self.ent_merge_out.pack(side="left",padx=5)
        ctk.CTkButton(row,text="選択",command=self.select_merge_save).pack(side="left",padx=5)

        ctk.CTkButton(frame,text="結合実行",command=self.run_merge).pack(pady=5)

        main=ctk.CTkFrame(frame);main.pack(fill="both",expand=True,padx=10,pady=10)

        # 左リスト
        self.txt_merge=ctk.CTkTextbox(main,width=260)
        self.txt_merge.pack(side="left",fill="y",padx=(0,8))

        # 右に開始/終了 表
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


    def select_merge_folder(self):
        d=filedialog.askdirectory()
        if not d:return
        self.lbl_merge_folder.configure(text=d)
        self.ent_merge_out.delete(0,"end");self.ent_merge_out.insert(0,d)

        self.csv_files=sorted(list(Path(d).glob("*.csv"))+list(Path(d).glob("*.txt")))
        self.txt_merge.delete("0.0","end")
        for w in self.inner_merge.winfo_children():w.destroy()

        for i,f in enumerate(self.csv_files):
            self.txt_merge.insert("end",f"{i+1:3d}  {f.name}\n")

            st,ed=get_start_end(f)
            row=ctk.CTkFrame(self.inner_merge);row.pack(fill="x",padx=3,pady=1)
            ctk.CTkLabel(row,text=f.name,width=240).pack(side="left")
            s=ctk.CTkEntry(row,width=200);s.insert(0,st);s.configure(state="readonly");s.pack(side="left")
            e=ctk.CTkEntry(row,width=200);e.insert(0,ed);e.configure(state="readonly");e.pack(side="left")


    def select_merge_save(self):
        d=filedialog.askdirectory()
        if d:self.ent_merge_out.delete(0,"end");self.ent_merge_out.insert(0,d)


    def run_merge(self):
        if not self.csv_files:return
        out=Path(self.ent_merge_out.get())/f"{self.ent_merge_name.get()}.csv"
        dfs=[read_data_auto(f) for f in self.csv_files]
        pd.concat(dfs,ignore_index=True).to_csv(out,index=False)
        ctk.CTkMessagebox(title="完了",message=str(out))



    # =========================================================
    # ■ ビット変換（MAP保存）
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
        json.dump({"map":f},open(CONFIG_FILE,"w"),indent=2)


    def select_bit_out(self):
        d=filedialog.askdirectory()
        if d:self.ent_bit_out.delete(0,"end");self.ent_bit_out.insert(0,d)


    def run_bit(self):
        if self.df_bit is None or self.df_map is None:
            return ctk.CTkMessagebox(title="エラー",message="CSV or MAP未選択")

        df=self.df_bit.copy()
        try:
            for bit_index,r in self.df_map.iterrows():
                o=str(r[0]).strip()
                n=str(r[1]).strip()
                if o not in df.columns: continue
                df[n]=df[o].apply(lambda x:(int(x)>>bit_index)&1)

            out=Path(self.ent_bit_out.get())/f"{Path(self.lbl_bit_csv.cget('text')).stem}_BIT.csv"
            df.to_csv(out,index=False)
            self.lbl_bit_state.configure(text="完了")
            ctk.CTkMessagebox(title="OK",message=str(out))

        except Exception as e:
            self.lbl_bit_state.configure(text="エラー")
            ctk.CTkMessagebox(title="ERROR",message=str(e))



# -----------------------------
# 実行
# -----------------------------
if __name__=="__main__":
    app=CSVApp()
    app.mainloop()

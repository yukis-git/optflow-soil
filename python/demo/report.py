"""summary.md of the demo (Japanese)."""
from dataclasses import replace

import numpy as np

from .field import (FootingField, GridField, _disc_masks, _green_lagrange_gmax, analytic_budget, band_fwhm, field_summary, prandtl_extent, prepeak_heave, second_order_area,
                    surface_budget)

FOOTING_MM = 30.0  # physical scale (an assumption, for interpretation only): the footing is 30 mm wide ...
D50_MM = 0.2  # ... and the sand's median grain size is 0.2 mm (Toyoura-sand-like), so 1 px = 0.2 mm at full size
PHI_DEG, PSI_PEAK_DEG = 45.0, 15.0  # the dense sand the angles of the mechanism are taken for: peak friction and dilation angle

# the sources the table cites (all of them verified by the literature check; "†": only the abstract / record was read)
REFERENCES = [
    ("Aiban & Znidarcic (1995)", "Aiban, S.A., Znidarcic, D. (1995) Centrifugal modeling of bearing capacity of shallow foundations on "
     "sands. Journal of Geotechnical Engineering (ASCE) 121(10), 704-712. †"),
    ("Arthur et al. (1977)", "Arthur, J.R.F., Dunstan, T., Al-Ani, Q.A.J., Assadi, A. (1977) Plastic deformation and failure in granular "
     "media. Géotechnique 27(1), 53-74. † (Vardoulakis 2005 を経由)"),
    ("Bolton (1986)", "Bolton, M.D. (1986) The strength and dilatancy of sands. Géotechnique 36(1), 65-78. † (係数は Strahler et al. 2016 を経由)"),
    ("Desrues & Andò (2015)", "Desrues, J., Andò, E. (2015) Strain localisation in granular media. Comptes Rendus Physique 16, 26-36."),
    ("Desrues et al. (1996)", "Desrues, J., Chambon, R., Mokni, M., Mazerolle, F. (1996) Void ratio evolution inside shear bands in "
     "triaxial sand specimens studied by computed tomography. Géotechnique 46(3), 529-546. †"),
    ("Diaz-Segura (2022)", "Diaz-Segura, E.G. (2022) Origen de la cuña de Prandtl: mecanismo de falla usado para cimentaciones "
     "superficiales. Gaceta Técnica 23(2). †"),
    ("Kimura et al. (1985)", "Kimura, T., Kusakabe, O., Saitoh, K. (1985) Geotechnical model tests of bearing capacity problems in a "
     "centrifuge. Géotechnique 35(1), 33-45. †"),
    ("Michalowski (1997)", "Michalowski, R.L. (1997) An estimate of the influence of soil weight on bearing capacity using limit "
     "analysis. Soils and Foundations 37(4), 57-64."),
    ("Oda & Kazama (1998)", "Oda, M., Kazama, H. (1998) Microstructure of shear bands and its relation to the mechanisms of dilatancy "
     "and failure of dense granular soils. Géotechnique 48(4), 465-481. †"),
    ("Rattez et al. (2022)", "Rattez, H. et al. (2022) Effect of grain size distribution on the shear band thickness evolution in "
     "sand. Géotechnique 72(4), 350-363."),
    ("Roscoe (1970)", "Roscoe, K.H. (1970) The influence of strains in soil mechanics. Géotechnique 20(2), 129-170. †"),
    ("Siddiquee et al. (1999)", "Siddiquee, M.S.A., Tanaka, T., Tatsuoka, F., Tani, K., Morimoto, T. (1999) Numerical simulation of "
     "bearing capacity characteristics of strip footing on sand. Soils and Foundations 39(4), 93-109."),
    ("Strahler et al. (2016)", "Strahler, A., Stuedlein, A.W., Arduino, P. (2016) Stress-strain response and dilatancy of sandy gravel "
     "in triaxial compression and plane strain. Journal of Geotechnical and Geoenvironmental Engineering 142(4), 04015098."),
    ("Tatsuoka et al. (1986)", "Tatsuoka, F., Sakamoto, M., Kawamura, T., Fukushima, S. (1986) Strength and deformation characteristics "
     "of sand in plane strain compression at extremely low pressures. Soils and Foundations 26(1), 65-84."),
    ("Tatsuoka et al. (1990)", "Tatsuoka, F., Nakamura, S., Huang, C.C., Tani, K. (1990) Strength anisotropy and shear band direction "
     "in plane strain tests of sand. Soils and Foundations 30(1), 35-54."),
    ("Tatsuoka et al. (1994)", "Tatsuoka, F., Siddiquee, M.S.A., Tanaka, T. (1994) Link among design, model tests, theories and sand "
     "properties in bearing capacity of footing on sand. Proc. 13th International Conference on Soil Mechanics and Foundation "
     "Engineering, New Delhi."),
    ("Tejchman & Herle (1999)", "Tejchman, J., Herle, I. (1999) A 'Class A' prediction of the bearing capacity of plane strain footings "
     "on sand. Soils and Foundations 39(5), 47-60."),
    ("Vardoulakis (2005)", "Vardoulakis, I. (2005) Localization in geomechanics. Proc. 16th International Conference on Soil "
     "Mechanics and Geotechnical Engineering, 3663ff."),
    ("Vermeer (1990)", "Vermeer, P.A. (1990) The orientation of shear bands in biaxial tests. Géotechnique 40(2), 223-236. †"),
]


def _fmt_s(x):
    return f"{x:.1f} s" if x >= 10 else f"{x:.2f} s"


def _mem(b):
    return "n/a" if not b else f"{b / 2**20:.0f} MB"


def _n(x, spec):
    """Number formatted with `spec`, or '-' where it is not finite (e.g. no valid pixel in the wedge)."""
    return "-" if x is None or not np.isfinite(x) else format(x, spec)


def _pct(x, total):
    return f"{100.0 * x / total:.0f} %"


def heave_facts(p):
    """The surface heave of the true field at the last frame, from the deformed ground surface (the same numbers as
    `field.surface_budget`): the highest point and the extent of the plateau beside the footing, the areas, and the
    volume budget (first-order closed form and the finite-displacement term)."""
    sb = surface_budget(p, p.s_final)
    bud = analytic_budget(p, p.s_final)
    # the passive wedge's plateau: beyond the pre-peak lobe (b + L_h) and before the mechanism's exit the surface is U_m alone
    x_w = 0.5 * (p.ue_half_support + p.xe)
    _, uy_w = FootingField(p, p.s_final).disp(np.array([p.xc + x_w]), np.array([p.y0]))
    return {**sb, "pre_peak": bud["pre_peak"], "dilation": bud["dilation"], "first_order": bud["total"],
            "second_order": second_order_area(p, p.s_final), "x_wedge": float(x_w), "h_wedge": float(-uy_w[0])}


def gmax_at(p, s, step=0.5):
    """gmax (Green-Lagrange, the tracker's definition) of the WHOLE field (pre-peak part + mechanism) at settlement s: the largest
    value under the footing's middle (|x'| < b / 2, y' > 0) and the largest inside the footing-edge discs (radius 3 w)."""
    Y, X = np.mgrid[p.y0 + 0.5 * step:p.y0 + p.depth:step, p.xc - p.half_box + 0.5 * step:p.xc + p.half_box:step]
    _, _, gr = GridField(p, X, Y).at(s)
    g = _green_lagrange_gmax(*gr)
    edge, _ = _disc_masks(p, X, Y)
    return float(g[(np.abs(X - p.xc) < 0.5 * p.b)].max()), float(g[edge].max())


def sp_sensitivity(p, fracs=(0.04, 0.03)):
    """What a smaller diffuse part would do (the same field with s_p / B = frac): the compression of the pre-peak part, the
    heave (gross and net of the trough) as a share of the intrusion, the strain of the mechanism, and when the mechanism
    carries half of the settlement."""
    out = []
    for f in fracs:
        q = replace(p, sp_frac=f)
        fs = field_summary(q, step=0.5)
        sb = surface_budget(q, q.s_final)
        out.append({"frac": f, "vol_pre_min": fs["vol_pre_min"], "heave_share": sb["heave_area"] / sb["intrusion_area"],
                    "net_share": sb["net_heave"] / sb["intrusion_area"], "gmax_p999": fs["gmax_p999"],
                    "vol_edge_max": fs["vol_edge_max"], "det_f_min": fs["det_f_min"], "half_s_over_B": half_mechanism_settlement(q) / q.B})
    return out


def half_mechanism_settlement(p):
    """The settlement at which the mechanism carries half of it, s_m(s) = s / 2 (bisection)."""
    lo, hi = 1e-9, 20.0 * p.sp
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if p.s_m(mid) < 0.5 * mid:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def physics_table(p):
    """Rows (item, this synthetic data, literature) of the table 'physical model and literature'.  Every number of the
    second column comes from the field itself.  The third column cites only sources that the literature check verified
    (see REFERENCES); 'by eye' marks values read from a figure, 'own computation' values computed from the closed forms
    of this model."""
    B = p.B
    px_mm = FOOTING_MM / B
    fs = field_summary(p, step=0.5)
    hv = heave_facts(p)
    fwhm = band_fwhm(p)
    ph = prepeak_heave(p)
    fs1 = field_summary(replace(p, ue_we=1.0), step=0.5)  # the same field with the edge band w_e = w, for comparison
    I = hv["intrusion_area"]
    gm_mid, gm_edge = gmax_at(p, p.sp)
    coulomb, roscoe, arthur = 45.0 + PHI_DEG / 2, 45.0 + PSI_PEAK_DEG / 2, 45.0 + (PHI_DEG + PSI_PEAK_DEG) / 4
    e30, e45 = prandtl_extent(30.0), prandtl_extent(45.0)
    alt = replace(p, alpha_deg=65.0, beta_deg=25.0)
    s_f, s_m_f = p.s_final, float(p.s_m(p.s_final))
    disp_fan = s_m_f * p.f_theta
    sens = (hv["first_order"] + hv["second_order"])
    lo, hi = hv["heave_plateau"]
    rows = [
        ("破壊機構と角度",
         f"運動学だけのモデル(応力も荷重もなく、ピーク荷重の判定もない。沈下の履歴 s(t) と s_p = {p.sp_frac:g} B は与えたもの)。"
         f"剛な楔(側面の角 α = {p.alpha_deg:g}°)+ 対数らせん扇形(中心はフーチングの縁、扇形の角 Θ = {p.theta_deg:g}°)+ "
         f"受働楔(外側のすべり線は水平から β' = {p.beta_deg:g}°、楔の内角 β = {p.beta_in_deg:g}°)。"
         f"すべり面は膨張角 ψ = {p.psi_deg:g}° で開く(速度の食い違いがすべり面と ψ の角をなす)。左右対称の一般せん断破壊。"
         f"α と β' は ψ_peak = {PSI_PEAK_DEG:g}° の方位の理論から取り、食い違いには経過平均の ψ = {p.psi_deg:g}° を使っている。"
         f"上界解でも、最適化した機構でも、Prandtl の解でもない",
         f"古典理論は、楔・対数らせん扇形・受働楔の Prandtl 型の機構で理想化する(Prandtl の解は重さのない土の解で (Michalowski 1997)、"
         f"もとは塑性体の硬さの論文で、土や基礎のためのものではない: Diaz-Segura 2022)。膨張角 ψ が摩擦角 φ より小さい非関連流れ則の材料では、"
         f"速度の食い違いは速度特性線(応力特性線から (φ − ψ)/2 傾く)に沿い、すべり面と ψ の角をなす (Michalowski 1997)。"
         f"すべり面は零伸びの線に一致する (Roscoe 1970)。扇形の速度 f0 exp(θ tan ψ) は、同じ流れ則からの自分の導出。"
         f"ψ < φ では極限解析の定理が成り立たないので上界とは言えない (Michalowski 1997)。"
         f"すべり面の向きの理論は 3 つあり、φ = {PHI_DEG:g}°、ψ = {PSI_PEAK_DEG:g}° では Coulomb {coulomb:.1f}°、Roscoe {roscoe:.1f}°、"
         f"Arthur–Vardoulakis {arthur:.0f}°(Vardoulakis 2005、Arthur et al. 1977 を経由)。粗い砂は Roscoe 型、細かい砂は Coulomb 型に近く、"
         f"分岐理論は間の角度を 1 つには決めない (Vermeer 1990)。ここでは中間の {p.alpha_deg:g}° を取った。"
         f"基礎の実験では、より急な角度が見られる: 豊浦砂の滑らかな基礎の下で帯は Coulomb の面 (69.4°) に一致し (Tatsuoka et al. 1990)、"
         f"SLB 砂の基礎の下では約 65〜67°(目で読んだ値: Tejchman & Herle 1999 の図 2b)。向きは今も論争中 (Tatsuoka et al. 1990)。"
         f"(α = 65°、β' = 25° にすると、頂点は {alt.ha / B:.2f} B、出口は {alt.xe / B:.1f} B、深さは {alt.depth_mech / B:.2f} B になる: 自分の計算)"),
        ("広がり",
         f"地表の出口は中心から {p.xe:.0f} px = {p.xe / B:.1f} B(フーチングの縁から {(p.xe - p.b) / B:.1f} B)、"
         f"機構の最大深さ {p.depth_mech:.0f} px = {p.depth_mech / B:.2f} B、楔の頂点の深さ {p.ha:.0f} px = {p.ha / B:.2f} B。"
         f"観察や計算で確かめた値ではなく、**様式化した大きさ**",
         f"地表のすべり面の出口までの距離を測った値は、確認できた文献にはなかった。読み取れる観察: 密な豊浦砂の基礎(幅 B0 = 50 cm、1 g)の"
         f"最大荷重時 (S/B0 = 4.8 %) には、縁の脇の狭い 2 本の帯だけで深さ約 0.5 B(目で読んだ値: Siddiquee et al. 1999 の図 15a)。"
         f"SLB 砂(基礎幅 10 cm)の s/B = 19 % では、頂点が基礎の底の約 1.15 B 下の V 字の帯と、その下から外向きに少なくとも中心から 1.9 B まで"
         f"続くほぼ水平な帯で、地表まで上る扇形や受働楔は写真に見えない(目で読んだ値: Tejchman & Herle 1999 の図 2b、論文の記号は b = 基礎幅)。"
         f"古典的な Prandtl の形を φ_peak に当てはめると、中心から出口まで φ = 30° で {e30[0]:.1f} B・深さ {e30[1]:.1f} B、"
         f"φ = 45° で {e45[0]:.1f} B・深さ {e45[1]:.1f} B(同じ閉じた式での自分の計算。角度は Michalowski 1997 の Prandtl 解)。"
         f"ただし古典理論(φ = 48° として)は B0 = 50 cm で N_γ を約 8 倍過大評価し (Tatsuoka et al. 1994)、この大きさの機構は密な砂の基礎の実験では"
         f"動員されていない。"
         f"ここの {p.xe / B:.1f} B / {p.depth_mech / B:.2f} B は Prandtl より小さい控えめな様式化で、観察で確かめた値ではない。"
         f"頂点の深さは、観察の約 1.15 B に対して {p.ha / B:.2f} B"),
        ("せん断帯の幅",
         f"ひずみ(gmax)の半値全幅 約 {fwhm:.0f} px(1 px = {px_mm:.2f} mm と仮定すると {fwhm * px_mm:.1f} mm = {fwhm * px_mm / D50_MM:.0f} D50、"
         f"D50 = {D50_MM:g} mm)。幅は一定で、軟化での狭まりはない。基礎幅は B/D50 = {B * px_mm / D50_MM:.0f}",
         "報告される厚さは D50 の 7〜20 倍で、砂と厚さの定義による。豊浦砂 12〜20 D50、SLB 砂 8.1〜9.0 D50(Siddiquee et al. 1999 の図 1 の説明)、"
         "基礎の下で約 20 D50 (Tatsuoka et al. 1994)、SLB 砂の基礎の下で約 10 D50 (Tejchman & Herle 1999)。三軸と DIC では、残留の厚さが"
         "一様な砂で約 10 D50 で、軟化の間に急速に細くなるので、ピーク付近はもっと広い (Rattez et al. 2022)。X 線と薄片では粒径の 7〜8 倍 "
         "(Oda & Kazama 1998)。ひずみ分布の半値全幅は、微視構造の帯の厚さとは別の定義。フーチング幅 30 mm、D50 = 0.2 mm は仮定。"
         "B/D50 = 150 は、粒径の影響が残る範囲で、帯が相対的に太く破壊が進行的になる (Tatsuoka et al. 1994; Tejchman & Herle 1999)"),
        ("帯の体積ひずみ",
         f"一定の膨張角 ψ = {p.psi_deg:g}° で、帯と扇形の体積ひずみ det F − 1 は最終フレームで最大 {fs['vol_band_max']:.2f}。"
         f"フーチングの縁の近傍(縁から {max(2 * p.rc, p.edge_radius * p.w):.0f} px = 3 w 以内)では特異点の正則化のため最大 {fs['vol_edge_max']:.2f}、"
         f"楔の頂点の近傍(約 7 w 以内)では最小 {fs['vol_apex_min']:.2f}(det F の最小は {fs['det_f_min']:.2f}、"
         f"x' = {int(round(fs['det_f_min_at'][0]))}, y' = {int(round(fs['det_f_min_at'][1]))} px)で、これらは正則化の見かけの値。"
         f"帯の脇では最小 {fs['vol_band_min']:.2f}(機構だけなら {fs['vol_band_min_mech']:.4f}。破壊前の流れのせん断と機構の帯のせん断の積の項で、"
         f"破壊前の縁の帯が狭いほど強い)。"
         f"ψ = {p.psi_deg:g}° は測定した値ではなく、帯の体積ひずみを 0.10 前後に抑えるために選んだ簡略化",
         f"密な砂の ψ_peak は 13〜20° と見積もられる: 平面ひずみの密な豊浦砂で応力・ダイレイタンシー関係の K ≈ 3.5 (Tatsuoka et al. 1986) から"
         f"(自分の計算)、φ − ψ = 28〜34° (Tatsuoka et al. 1990) から、平面ひずみで φ_max − φ_cv = 0.8 ψ_max (Bolton 1986。係数は Strahler et al. 2016 を経由。"
         f"経験式で約 2° のばらつきがあり、30 mm の 1 g の模型のような低い応力ではあまり当てにならない)。"
         f"CT では帯の中の間隙比が限界値で頭打ちになり (Desrues et al. 1996; Desrues & Andò 2015)、局所の間隙比が標準の最大間隙比を超えうる "
         f"(Oda & Kazama 1998)。**帯の体積ひずみそのものの測定値は見つからなかった**。ここの 0.09〜{fs['vol_band_max']:.2f} は間隙比の頭打ちとして無理はないが、"
         f"較正したものではなく、実際の帯がそこまで膨張するとは言えない。ψ は経過平均で一定(ピーク付近の強い膨張と、その後の限界状態は表現しない)"),
        ("隆起と体積収支",
         f"最終フレームで、フーチングの外側の地表は縁の脇の峰で最大 {hv['heave_max']:.1f} px 隆起する(x' = {lo:.0f}〜{hi:.0f} px、"
         f"縁から {(lo - p.b) / B:.1f}〜{(hi - p.b) / B:.1f} B。破壊前の流れの隆起と受働楔の持ち上げが重なった峰)。"
         f"さらに外側(x' = {hv['x_wedge']:.0f} px)は受働楔が平行移動した平らな台地で {hv['h_wedge']:.1f} px。"
         f"フーチングの縁では地表は初期より {abs(hv['h_edge']):.1f} px 低く"
         f"(フーチングの沈下 {s_f:g} px に対し、食い込みは {s_f + hv['h_edge']:.1f} px)、初期の高さに戻るのは x' = {hv['zero_cross']:.0f} px"
         f"(縁から {(hv['zero_cross'] - p.b) / B:.2f} B)。縁の脇の沈下部は、縁に引きずられた土と正則化した縁による(面積 {hv['trough_area']:.0f} px²。"
         f"くぼみではなく、縁の近くの狭い範囲)。フーチングの外側で、隆起部の面積 {hv['heave_area']:.0f} px²"
         f"(押し込んだ面積 B·s = {I:.0f} px² の {_pct(hv['heave_area'], I)})、脇の沈下部 {hv['trough_area']:.0f} px²、"
         f"差し引き(正味の隆起){hv['net_heave']:.0f} px²({_pct(hv['net_heave'], I)})。土の面積の正味の変化は {hv['net_dv']:+.0f} px²"
         f"(膨張): 破壊前の部分の圧縮 {hv['pre_peak']:.0f} px²(押し込んだ面積の {_pct(-hv['pre_peak'], I)}。破壊前の部分の隆起は"
         f"(1 − κ)B·s_e の {100 * ph['net_outside'] / ((1 - p.ue_kappa) * B):.1f} %)を、帯と扇形の膨張 {hv['dilation']:+.0f} px²"
         f"(閉じた式 2 D s_m、D = {p.dilation:.1f} px²/px)が上回る。有限変位の 2 次項 {hv['second_order']:.0f} px²"
         f"(形を固定した機構を s_m に比例させたことによる)を加えた {sens:.0f} px² は、面積の変化にほぼ一致する"
         f"(差 {abs(sens - hv['net_dv']):.0f} px²、1 次の閉じた式の残差)。"
         f"次の関係は定義から従う簿記上の等式で、検証ではない: フーチングの外側の正味の隆起 = B·s + 土の面積の変化 − 縁で土がフーチングの底面より上に"
         f"はみ出して重なる面積({hv['contact_overlap']:.0f} px²、押し込んだ面積の {_pct(hv['contact_overlap'], I)}、縁を正則化した帯の幅程度の範囲。"
         f"正則化の見かけの量)",
         "押し込まれた体積は、土の圧縮と膨張を差し引いて地表の隆起になる、という釣り合いは、運動学的に許される場については厳密に成り立つ。"
         "しかし隆起と沈下の比や隆起の形を測った値は、確認できた文献の中に見つからなかった。したがって、ここの隆起の量と形は実験で確かめていない"
         f"(破壊前の流れが押し込んだ体積の (1 − κ) = {100 * (1 - p.ue_kappa):.0f} % を基礎の脇の隆起として返すのは、"
         "密な砂の破壊前の収縮が小さいこと (Tatsuoka et al. 1986) に合わせた設計上の選択で、測定ではない)。"
         "定性的には、s/B = 19 % で地表近くの基礎の脇に大きなひずみが集まる (Tejchman & Herle 1999 の図 2b、目で読んだ値)"),
        ("破壊前の圧縮",
         f"s_p = {p.sp_frac:g} B = {p.sp:.1f} px で飽和する拡散した流れで、**ほぼ非圧縮**(圧縮は、破壊前の部分が押し込む面積 B·s_e の κ = {p.ue_kappa:g} だけ)。"
         f"**滑らかな沈下の立ち上がりと縁の脇の隆起のための道具**で、較正した場ではない。"
         f"流れ関数 Φ = Ψ(x') K(y') の流れ U_iso(重み 1 − κ。1 次の発散は丸め誤差まで 0: 最大 {fs['div_iso_max']:.0e})と、"
         f"基礎の下の圧縮の塊 U_c(重み κ。基礎の底では U_iso と合わせて土がフーチングと同じ量だけ動く)の和。"
         f"土は基礎の下で一緒に沈み、深さとともに外へ広がり(深さ {p.ue_depth_k:.0f} px = {p.ue_lk:g} B で 0 に戻る)、"
         f"基礎の脇で幅 {p.ue_lh:g} B(中心から {p.ue_half_support:.0f} px)の隆起に戻る。縁には半幅 w_e = {p.ue_edge_hw:.0f} px = {p.ue_we:g} w "
         f"の帯があり、鉛直の流れが下向きから上向きに変わってひずみが集中する。"
         f"体積ひずみ(det F − 1、真の行列式)は {fs['vol_pre_min']:+.4f} 〜 {fs['vol_pre_max']:+.4f}、土全体の平均 {fs['vol_pre_mean']:+.5f}。"
         f"最小(圧縮 {abs(fs['vol_pre_min']) * 100:.2f} %)は基礎の下の深さ y' = {fs['vol_pre_min_at'][1]:.0f} px で、U_c の圧縮と流れの 2 次項。"
         f"最大(+{fs['vol_pre_max'] * 100:.2f} %)は縁の脇 (x', y') = ({abs(fs['vol_pre_max_at'][0]):.0f}, {fs['vol_pre_max_at'][1]:.0f}) px で、"
         f"1 次の発散ではなく、縁の帯のせん断と深さ方向の流れの変化の積の 2 次項(帯の幅に反比例: w_e = w なら最大 {fs1['vol_pre_max']:+.3f})。"
         f"−0.5 % より強く圧縮される土は {100 * fs['vol_pre_share_below']:.0f} %、+0.2 % を超えるのは土の面積の {100 * fs['vol_pre_share_above']:.1f} %"
         f"(縁の脇の縦長の帯)、+0.5 % を超えるのは {100 * fs['vol_pre_share_above5']:.1f} %。"
         f"この部分だけの gmax は、基礎の下の中央で最大 {fs['gmax_pre_middle_max']:.2f}、縁の帯で最大 {fs['gmax_pre_max']:.2f}",
         f"要素試験では、密な豊浦砂の平面ひずみ圧縮の初期の収縮は、拘束圧 σ3' = 4.9 kPa で約 0.1 %、392 kPa で 0.5 % 未満"
         f"(目で読んだ値: Tatsuoka et al. 1986 の図 6a, 7a)。つまり 0.1〜0.5 % の下端は低い拘束圧、上端は高い拘束圧の値で、"
         f"30 mm の 1 g の模型基礎の下の応力(数 kPa)は下端の側。ここの最大の圧縮 {100 * abs(fs['vol_pre_min']):.2f} % は上端(392 kPa)の値に当たり、"
         f"低い応力の 0.1 % よりは数倍大きい。κ = {p.ue_kappa:g} は、この圧縮を要素試験の範囲の上端に収める上限として選んだ。"
         f"密な砂はせん断帯ができる前にダイレイタンシーに転じる (Tejchman & Herle 1999。有限要素の計算結果で、測定ではない)が、"
         f"ここの破壊前の部分は膨張しない(「モデル化していないもの」)。"
         f"ひずみの比較は、沈下をそろえて行う。Tejchman & Herle 1999 の図 2b の s/b = 6 %(論文の b = 基礎幅)は、ここでは s = {0.06 * B:.0f} px = s_p で、"
         f"拡散した部分は s_e = s_p だが、機構の部分も s_m = {float(p.s_m(p.sp)):.1f} px とゼロではない。"
         f"その図の V 字の領域は横に ±1.5 B、深さ約 1.5 B、せん断ひずみ 3〜9 %、縁に最大 15〜18 %(目で読んだ値)。"
         f"ここの全体の場の gmax は s = s_p で、基礎の下の中央 (|x'| < b/2) で最大 {gm_mid:.2f}(拡散した部分だけなら {fs['gmax_pre_middle_max']:.2f}。"
         f"違いは楔の頂点の機構の分)、縁の近傍で最大 {gm_edge:.2f} で、3〜9 % の上限の約 {gm_mid / 0.09:.0f} 倍、15〜18 % の約 {gm_edge / 0.165:.0f} 倍。"
         f"領域の広がりは、横に {p.ue_half_support / B:.1f} B、深さ {p.ue_lk:.1f} B(それぞれ約 {p.ue_half_support / (1.5 * B):.1f} 倍、{p.ue_lk / 1.5:.1f} 倍)。"
         f"縁の帯の幅 w_e を w ではなく {p.ue_we:g} w にしたのは、w_e = w だと 2 次項が最大 {fs1['vol_pre_max']:+.3f}、拡散した部分の縁の帯の gmax が {fs1['gmax_pre_max']:.2f}、"
         f"機構の帯との干渉で帯の脇の体積ひずみが最小 {fs1['vol_band_min']:.2f}(ここは {fs['vol_band_min']:.2f})になるため。"
         f"最大荷重時には縁の脇の狭い 2 本の帯だけ: Siddiquee et al. 1999; Tatsuoka et al. 1994)"),
        ("相対沈下 s/B",
         f"最終フレームで s/B = {p.s_final / B:.2f}(沈下 {p.s_final:g} px、B = {B:g} px)。拡散した部分の尺度は s_p = {p.sp_frac:g} B。"
         f"s = s_p のとき機構は沈下の {100 * float(p.s_m(p.sp)) / p.sp:.0f} % を担い、半分を超えるのは s/B = {half_mechanism_settlement(p) / B:.2f} 以降"
         f"(s_m = s − s_p tanh(s/s_p))",
         "最大荷重の沈下は、文献で約 3〜14 %(基礎の大きさ、粒径、応力レベルによる): 豊浦砂(B0 = 50 cm、1 g)で S/B0 = 4.8 %、"
         "有限要素で 3〜5 % (Siddiquee et al. 1999)、SLB 砂で基礎幅 10 cm〜1 cm に対して s/B ≈ 8〜14 %(目で読んだ値: Tejchman & Herle 1999)で、"
         "基礎が小さいほど大きい。最終の s/B = 0.2 はピークをかなり過ぎた状態。V 字の帯がはっきり現れるのは s/B ≈ 19 % (Tejchman & Herle 1999)。"
         "s_p = 0.06 B は、この範囲の代表値"),
        ("モデル化していないもの",
         f"左右非対称な片側すべり、**破壊前の膨張**(密な砂はせん断帯ができる前に膨張に転じるが、破壊前の部分 U_e はほぼ非圧縮で、膨張せず、"
         f"κ の分だけ圧縮する)、破壊前の流れの形は固定で振幅 s_e だけが変わる(縁の帯の幅 w_e = {p.ue_we:g} w や広がり {p.ue_lh:g} B・{p.ue_lk:g} B は"
         f"設計上の値で、測定ではない)、基礎の脇の隆起の割合(押し込んだ体積の (1 − κ) = {100 * (1 - p.ue_kappa):.0f} % を返すのは仮定)、"
         f"圧縮の塊 U_c(κ を逃がすための道具)、ひずみ軟化と限界状態でのダイレイタンシーの飽和(ψ は一定)、帯の幅の変化(一定)、粒子破砕、"
         f"機構の幾何の移動(形を固定した機構を s_m に比例させる。扇形の点は E まわりの剛体回転に比べ (s_m f)²/(2r) だけずれる: "
         f"r = {0.5 * B:.0f} px で約 {disp_fan ** 2 / (2 * 0.5 * B):.1f} px、r = {B:.0f} px で約 {disp_fan ** 2 / (2 * B):.1f} px。"
         f"Green–Lagrange ひずみには r = {0.5 * B:.0f} px で約 {0.5 * (disp_fan / (0.5 * B)) ** 2:.3f} の見かけの成分が付く)、"
         f"粗い基礎(楔は真下にだけ動く。滑らかな基礎では帯が底の 1 点から出る)、応力レベルによる φ と ψ の変化、"
         f"フーチングの縁で土が底面より上にはみ出す重なり({hv['contact_overlap']:.0f} px²)、容器の壁の摩擦",
         "片側すべり: 遠心模型の密な砂の帯基礎で破壊は常に片側に起きた (Aiban & Znidarcic 1995。偏心・傾斜・埋め込みの実験で、中心載荷だけでの確認ではない)。"
         "基礎が大きいほど破壊は進行的 (Kimura et al. 1985; Tatsuoka et al. 1994)。限界状態 (Desrues & Andò 2015)。軟化での帯の狭まり (Rattez et al. 2022)。"
         "粒径の影響は、基礎幅/D50 が豊浦砂で約 500、SLB 砂で約 100 より小さいときに残る (Tejchman & Herle 1999)。"
         "滑らかな基礎の下の帯は底の 1 点から出る (Tatsuoka et al. 1990)。粒子破砕は、確認できた文献がないので出典なし"),
    ]
    return rows


def sp_sensitivity_text(p):
    """The paragraph on the size of the diffuse part (s_p), with the sensitivity computed from the field."""
    hv = heave_facts(p)
    fs = field_summary(p, step=0.5)
    I = hv["intrusion_area"]
    sens = sp_sensitivity(p)
    parts = "。".join(
        f"s_p = {r['frac']:g} B では、圧縮の最小 {100 * r['vol_pre_min']:.2f} %、隆起は総量で押し込んだ面積の {100 * r['heave_share']:.0f} %、"
        f"正味で {100 * r['net_share']:.0f} %、gmax の 99.9 パーセンタイル {r['gmax_p999']:.2f}、フーチングの縁の近傍の体積ひずみの最大 {r['vol_edge_max']:.2f}、"
        f"機構が沈下の半分を担うのは s/B = {r['half_s_over_B']:.2f} 以降" for r in sens)
    return (f"**拡散した部分の大きさ(s_p)について**: 破壊前の流れはほぼ非圧縮で、フーチングが押し込む面積のうち圧縮として吸収するのは "
            f"{_pct(-hv['pre_peak'], I)}(破壊前の部分が押し込む面積 B·s_e の {100 * p.ue_kappa:.0f} %)だけ(残りは基礎の脇の隆起として返す)。s_p = {p.sp_frac:g} B では、隆起は総量で押し込んだ面積の "
            f"{_pct(hv['heave_area'], I)}、正味で {_pct(hv['net_heave'], I)}(圧縮の最小 {100 * fs['vol_pre_min']:.2f} %、"
            f"機構が半分を担うのは s/B = {half_mechanism_settlement(p) / p.B:.2f} 以降)。"
            f"s_p を小さくしても圧縮・隆起の釣り合いはほとんど変わらず(破壊前の流れの圧縮が元から小さいため)、機構の分担が増えてひずみが強くなり、"
            f"機構が育つ時期も早まる: {parts}。"
            f"V 字の帯が s/B ≈ 19 % でようやくはっきりする観察 (Tejchman & Herle 1999) と、ひずみを強くしすぎないことの、どちらを優先するかは"
            f"判断が分かれる。ここでは前者を取って s_p = {p.sp_frac:g} B を残した。変える場合は `Params.sp_frac` で、全体を再計算する。")


def deviation_text(p):
    """The deviations of the pre-peak part from the design brief, with the measured numbers (all computed from the field)."""
    fs = field_summary(p, step=0.5)
    f1 = field_summary(replace(p, ue_we=1.0), step=0.5)
    k1 = field_summary(replace(p, ue_kappa=0.1), step=0.5)
    k2 = field_summary(replace(p, ue_kappa=0.2), step=0.5)
    return (f"**設計仕様(ブリーフ)からの逸脱**(破壊前の部分 U_e。いずれも測定した値で、了承済み): "
            f"(1) 破壊前の部分 s_p U_e の体積ひずみ det F − 1(真の行列式)は [{fs['vol_pre_min']:+.4f}, {fs['vol_pre_max']:+.4f}](0.5 px の格子。0.25 px の格子では −0.0049、+0.0122)で、"
            f"仕様の [−0.005, +0.002](縁の近傍 [−0.01, +0.005])には収まらない。最小は基礎の下 (x', y') = ({fs['vol_pre_min_at'][0]:.0f}, {fs['vol_pre_min_at'][1]:.0f}) px、"
            f"最大は縁の脇 ({fs['vol_pre_max_at'][0]:.0f}, {fs['vol_pre_max_at'][1]:.0f}) px(縁の近傍の外。近傍の中の最大は {fs['vol_pre_edge_max']:+.4f})。"
            f"1 次の発散は至る所 0 以下なので、正の部分は縁の帯のせん断と深さ方向の流れの変化の積の 2 次項だけで、"
            f"+0.2 % を超えるのは土の面積の {100 * fs['vol_pre_share_above']:.1f} %、+0.5 % を超えるのは {100 * fs['vol_pre_share_above5']:.1f} %。"
            f"縁の帯を持つ流れ関数の流れでは、+0.002 は帯の半幅が 100 px 超でないと届かない。"
            f"(2) 縁の帯の半幅は w ほどではなく w_e = {p.ue_we:g} w = {p.ue_edge_hw:.0f} px。w_e = w だと最大 {f1['vol_pre_max']:+.3f}、縁の帯の gmax {f1['gmax_pre_max']:.2f}、"
            f"帯の脇の体積ひずみの最小 {f1['vol_band_min']:.2f}(ここは {fs['vol_band_min']:.2f})。"
            f"(3) κ は 0.1〜0.2 ではなく {p.ue_kappa:g}: 最小が −0.005 を下回らない最大の値。κ = 0.1 では最小 {k1['vol_pre_min']:+.4f}、κ = 0.2 では {k2['vol_pre_min']:+.4f}"
            f"(最大は {k1['vol_pre_max']:+.4f}、{k2['vol_pre_max']:+.4f} と、ゆっくりしか下がらない)。")


def reference_lines():
    return [f"- {text}" for _, text in REFERENCES]


def summary_md(scene, ev, metrics, args, out):
    """Text of summary.md: what was done, parameters, tables of settlement and accuracy, run times, where and why the
    errors are larger, how to run again.  All numbers come from `metrics` and the evaluation `ev`."""
    p = scene.params
    rows = ev.rows
    S = metrics["summary"]
    st = metrics["stage_seconds"]
    cli = metrics["tracker"]
    cfg = metrics["tracker_config"]
    algo = metrics["tracker_algo_seconds"]
    n = scene.n_frames
    last = rows[-1]
    first = rows[0]
    sel = metrics["selected_frames"]
    crop = metrics["crop"]
    cmd = "python -m demo.run_demo" + (f" --size {args.size}" if args.size != "full" else "")
    if args.frames:
        cmd += f" --frames {args.frames}"
    if args.settlement is not None:
        cmd += f" --settlement {args.settlement:g}"
    if args.impl != "rust":
        cmd += f" --impl {args.impl}"
    fwhm = band_fwhm(p)
    peak_ratio = S["gmax_peak_est_final"] / S["gmax_peak_true_final"] if S["gmax_peak_true_final"] else float("nan")
    peak_ratio_all = S["gmax_peak_est_final"] / S["gmax_peak_true_all_final"] if S["gmax_peak_true_all_final"] else float("nan")
    band_ratio = last["disp_rms_band"] / last["disp_rms_out"] if last["disp_rms_out"] else float("nan")
    w_px, h_px = scene.width, scene.height
    fsum = field_summary(p, step=0.5)
    wedge_lost = not np.isfinite(last["settlement_est"])
    L = []
    a = L.append
    a("# フーチング沈下の合成モデル試験: 追跡デモの結果\n")
    a(f"- 実行日時: {metrics['date']}")
    a(f"- コマンド: `{cmd}`")
    a(f"- 追跡: {'Rust CLI (`optflow seq`)' if args.impl == 'rust' else 'Python 参照実装 (`python -m optflow_ref seq`)'}"
      f"、実行したコマンド: `{cli['command']}`\n")

    a("## 1. 何をしたか\n")
    a(f"1. **合成動画の生成** ({st.get('generate', float('nan')):.1f} 秒): {w_px}×{h_px} px、{n} フレーム、{scene.fps:g} fps、"
      f"H.264 (crf 12)。砂状のランダムなスペックルの土を入れた土槽、動かない容器の壁と背景(壁に固定マーカ)、"
      f"地表中央から一定速度で最大 {p.s_final:g} px 沈下する剛なフーチング(幅 {2 * p.b:g} px)。"
      f"土は解析的な変位場に従って動く。変位場は、剛な楔・対数らせん扇形・受働楔からなる破壊機構(膨張角 ψ を持つ、運動学的に許される機構)に、"
      f"破壊前のほぼ非圧縮の拡散した流れ(基礎の脇の隆起と縁のひずみ集中を持つ)を重ねたもの(「3. 物理モデルと文献との対応」)。運動学だけの様式化した場で、実験で較正したものではない。"
      f"逆写像で各フレームを描画しているので、真値(変位とひずみ)は解析的に分かる。"
      f"カメラノイズ(標準偏差 {scene.noise * 100:.1f} %)と照明のゆらぎ(フレームごとに ±{scene.flicker * 100:.1f} %)を加えてある。")
    a(f"2. **ROI**: 最初のフレームで、容器の中の土だけを囲む多角形 (`roi.json`)。壁・背景・フーチングは含めない。"
      f"土の初期位置を追うので、土が動いても ROI の画素は同じ土の粒子を指す(結果はフレーム 0 の格子上)。")
    a(f"3. **追跡**: `{cli['command'].split(' ', 1)[1] if args.impl == 'rust' else cli['command']}`。"
      f"自動クロップ(クロップ領域 x {crop['x0']}–{crop['x1']}, y {crop['y0']}–{crop['y1']} px)、"
      f"参照フレームは平均信頼度が {cfg['ref_update_conf']:g} を下回ると自動で更新される"
      f"(この実行での更新回数: {S['reference_updates']} 回)。")
    a("4. **真値との比較**: フレーム 0 の格子上で、ROI の有効画素について変位とひずみ(Green-Lagrange ひずみから gmax と体積ひずみ)の"
      "誤差を計算。沈下量は「フーチング直下の楔の中の画素の u_y の中央値」で推定し、同じ画素の真値の中央値と比べる。")
    a("5. **図と動画**: `overview.png`(選んだフレームの推定・真値・誤差、色のスケールは共通)、`curves.png`(フレームごとの精度)、"
      "`strain_animation.mp4`(入力動画と、土の粒子の**現在の位置**に描いた gmax の重ね合わせを並べたもの)。\n")

    a("## 2. 設定\n")
    a("| 項目 | 値 |")
    a("|---|---|")
    a(f"| 画像 / フレーム数 | {w_px}×{h_px} px / {n} フレーム (フレーム 0 が参照) |")
    a(f"| 沈下量 | 0 → {p.s_final:g} px(一定速度、1 フレームあたり {p.s_final / (n - 1):.2f} px、s/B = {p.s_final / p.B:.2f}) |")
    a(f"| 土槽 | 内幅 {2 * p.half_box:g} px、深さ {p.depth:g} px、地表 y = {p.y0:g} px。機構の出口は壁から {p.half_box - p.xe:.0f} px、"
      f"底は機構の最大深さの {(p.depth - p.depth_mech) / p.B:.1f} B 下 |")
    a(f"| フーチング | 幅 B = {p.B:g} px(半幅 b = {p.b:g} px)、剛体 |")
    a(f"| 破壊機構 | 楔の側面の角 α = {p.alpha_deg:g}°、外側のすべり線の角 β' = {p.beta_deg:g}°、膨張角 ψ = {p.psi_deg:g}°、"
      f"β = {p.beta_in_deg:g}°、扇形の角 Θ = {p.theta_deg:g}° |")
    a(f"| 機構の大きさ | 楔の頂点の深さ {p.ha:.0f} px、最大深さ {p.depth_mech:.0f} px、地表に達する位置は中心から {p.xe:.0f} px |")
    a(f"| せん断帯の幅 | 遷移の半幅 w = {p.w:g} px(実測したひずみの半値全幅 ≈ {fwhm:.0f} px)。フーチングの縁の近傍(半径 {2 * p.rc:g} px)で扇形を絞り、"
      f"縁({p.edge_radius * p.w:g} px 以内)と楔の頂点({p.apex_radius * p.w:g} px 以内)では帯を広げる(特異点の正則化。縁の影響は縁から {max(2 * p.rc, p.edge_radius * p.w):g} px まで及ぶ) |")
    a(f"| 破壊前の部分 | s_p = {p.sp_frac:g} B = {p.sp:.1f} px。流れ関数の流れ(重み 1 − κ)と圧縮の塊(重み κ = {p.ue_kappa:g})の和。"
      f"流れの深さ L_k = {p.ue_depth_k:.0f} px、基礎の脇の隆起の裾は中心から {p.ue_half_support:.0f} px(L_h = {p.ue_lobe:.0f} px)、"
      f"縁の帯の半幅 w_e = {p.ue_edge_hw:.0f} px。圧縮の塊の深さ L_c = {p.ue_comp_depth:.0f} px |")
    a(f"| 時間変化 | 沈下 s = s_e + s_m、s_e = s_p tanh(s/s_p)(拡散した破壊前の流れ)、s_m = s − s_e(機構)。帯は s_m ∝ s³ で徐々に現れる |")
    tf = ev.truth.frame(float(p.s_final))
    a(f"| 真値の最大せん断ひずみ(最終) | 土の全域で 99.9 パーセンタイル {fsum['gmax_p999']:.2f}、最大 {fsum['gmax_max']:.2f}"
      f"(Green-Lagrange ひずみの主ひずみの差。追跡が有効でガード外の画素だけでは 99.9 パーセンタイル {S['gmax_peak_true_final']:.2f}) |")
    a(f"| ROI | 多角形(矩形)x {scene.roi_bbox()[0]}–{scene.roi_bbox()[2]}, y {scene.roi_bbox()[1]}–{scene.roi_bbox()[3]} px、"
      f"壁・底から約 {max(4.0, 0.012 * scene.width):.0f} px、地表の {max(3.0, 0.02 * scene.height):.0f} px 下から |")
    a(f"| トラッカー設定 | 既定: 格子 {cfg['grid_step']} px、POC 窓 {cfg['poc_win']}、max_disp {cfg['max_disp']:g}、"
      f"波長 {cfg['lam']:g} px ×{cfg['scales']}、σ_w {cfg['sigma_w']:g}、ひずみ窓 {cfg['strain_win']}、"
      f"端のガード {cfg['edge_guard_px']:.1f} px、conf_min {cfg['conf_min']:g}" +
      (f"、追加の設定 {', '.join(args.set)}" if args.set else "") + " |")
    a("")

    a("## 3. 物理モデルと文献との対応\n")
    a(f"真値の変位場は、破壊が進むにつれて機構が育つ形で作った(`python/demo/field.py`)。運動学だけのモデルで、様式化してあり、"
      f"実験で較正してはいない。文献の欄は、読めた出典だけに基づく: 「目で読んだ値」は図から読んだおおよその値、「自分の計算」はこのデモの閉じた式から"
      f"計算した値。実験で確かめていないものは、その旨を書いた。物理的な大きさは、フーチング幅 {FOOTING_MM:g} mm、砂の D50 = {D50_MM:g} mm"
      f"(豊浦砂程度)、すなわち 1 px = {FOOTING_MM / p.B:.2f} mm と**仮定した**場合の解釈のためだけのもの。\n")
    a("| 項目 | この合成データ | 文献の値・観察 |")
    a("|---|---|---|")
    for item, mine, lit in physics_table(p):
        a(f"| {item} | {mine} | {lit} |")
    a("")
    a(sp_sensitivity_text(p) + "\n")
    a(deviation_text(p) + "\n")
    a("参考文献(「†」は要旨・書誌だけを確認したもの):\n")
    L.extend(reference_lines())
    a("")

    a("## 4. 結果の要点\n")
    if not wedge_lost:
        a(f"- **沈下量**: 最終フレームで、フーチングの沈下 {S['settlement_true_final']:.2f} px、楔の中の土の真値(推定に使う画素の u_y の中央値)"
          f"{S['settlement_ref_final']:.2f} px、推定 {S['settlement_est_final']:.3f} px(推定 − 土の真値 = {last['settlement_err']:+.3f} px)。"
          f"全フレームでの誤差の最大は {S['settlement_err_max_abs']:.3f} px。土の真値がフーチングの沈下より小さいのは、破壊前の流れの鉛直変位が"
          f"深さとともに減る(楔の中の土は、深いほど、フーチングの沈下から s_e の一部だけ遅れる)ため。")
    else:
        kv = S["settlement_last_valid_frame"]
        tail = (f"最後に推定が得られたフレーム {kv} では、フーチングの沈下 {S['settlement_true_last_valid']:.2f} px、土の真値 "
                f"{S['settlement_ref_last_valid']:.2f} px、推定 {S['settlement_est_last_valid']:.3f} px。全フレームでの |推定 − 土の真値| の最大は "
                f"{S['settlement_err_max_abs']:.3f} px。" if kv is not None else "どのフレームでも推定は得られなかった。")
        a(f"- **沈下量**: 最終フレームでは楔の画素がすべて無効(強い帯が狭い楔を覆う)で、沈下の推定は得られない。{tail}")
    a(f"- **変位誤差**(有効画素、フレーム 0 の格子): 最終フレームの RMS {S['disp_rms_final']:.4f} px、99 パーセンタイル "
      f"{S['disp_p99_final']:.3f} px、最大 {S['disp_max_final']:.3f} px。最悪フレームは {S['disp_rms_worst_frame']} で RMS "
      f"{S['disp_rms_worst']:.4f} px。")
    a(f"- **有効画素の割合**: ROI に対して最小 {S['valid_frac_min']:.3f}。ROI の縁から端のガード距離({cfg['edge_guard_px']:.1f} px)以内は"
      f"設計上無効になるので、その外側の画素(ROI の {S['interior_pixels'] / S['roi_pixels']:.3f})に対する割合は最小 "
      f"{S['valid_frac_eff_min']:.3f}。")
    a(f"- **せん断帯**(真値の最終フレームの gmax が最大の {int(100 * 0.4)} % 以上の画素 {S['band_pixels']} 個): 最終フレームの変位誤差 RMS は"
      f"帯の中 {last['disp_rms_band']:.3f} px、帯の外 {last['disp_rms_out']:.3f} px(比 {band_ratio:.1f})。gmax の誤差 RMS は"
      f"帯の中 {last['gmax_rms_band']:.3f}、外 {last['gmax_rms_out']:.3f}。gmax の 99.9 パーセンタイルは、有効画素で推定 "
      f"{S['gmax_peak_est_final']:.3f}、真値 {S['gmax_peak_true_final']:.3f}(比 {peak_ratio:.2f})。最もひずみの強い画素は無効になるので、"
      f"ガード外の全画素の真値の 99.9 パーセンタイル {S['gmax_peak_true_all_final']:.3f} に対する比は {peak_ratio_all:.2f}。")
    a(f"- **参照フレームの更新**: {S['reference_updates']} 回。")
    if cli.get("capture_warning") or cli.get("crop_warning"):
        a(f"- **注意**: トラッカーが警告を出した: `{cli['stderr']}`")
    else:
        a("- トラッカーの警告(捕捉範囲、クロップ境界)はなかった。")
    a("")

    a("## 5. 沈下曲線(真値と推定)\n")
    a("![settlement curve](curves.png)\n")
    a("| フレーム | フーチングの沈下 [px] | 楔の中の土の真値 [px] | 推定 [px] | 推定 − 土の真値 [px] | 変位誤差 RMS [px] | 有効率 |")
    a("|---:|---:|---:|---:|---:|---:|---:|")
    ks = sorted({int(round(x)) for x in np.linspace(1, n - 1, min(12, n - 1))})
    for k in ks:
        r = rows[k - 1]
        a(f"| {k} | {r['settlement_true']:.2f} | {_n(r['settlement_ref'], '.3f')} | {_n(r['settlement_est'], '.3f')} | "
          f"{_n(r['settlement_err'], '+.3f')} | {r['disp_rms']:.4f} | {r['valid_frac']:.3f} |")
    a("")

    a("## 6. 精度\n")
    a("選んだフレーム(overview.png のフレーム)の変位誤差:\n")
    a("| フレーム | 沈下 [px] | RMS [px] | p99 [px] | 最大 [px] | 帯の中 RMS | 帯の外 RMS | 有効率(ROI) | 有効率(ガード外) |")
    a("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in sel:
        r = rows[k - 1]
        a(f"| {k} | {r['settlement_true']:.1f} | {r['disp_rms']:.4f} | {r['disp_p99']:.3f} | {r['disp_max']:.3f} | "
          f"{r['disp_rms_band']:.3f} | {r['disp_rms_out']:.3f} | {r['valid_frac']:.3f} | {r['valid_frac_eff']:.3f} |")
    a("")
    a("ひずみの誤差 RMS(推定 − 真値)、せん断帯の中 / 外:\n")
    a("| フレーム | " + " | ".join(f"{k} 中 | {k} 外" for k in ("Exx", "Eyy", "Exy", "gmax", "vol")) + " |")
    a("|---:|" + "---:|---:|" * 5)
    for k in sel:
        r = rows[k - 1]
        a(f"| {k} | " + " | ".join(f"{r[f'{q}_rms_band']:.3f} | {r[f'{q}_rms_out']:.3f}" for q in ("Exx", "Eyy", "Exy", "gmax", "vol")) + " |")
    a("")
    a("![overview](overview.png)\n")
    a("（`overview.png`: 列がフレーム、行が u_x, u_y の推定、変位誤差、gmax の推定・真値・誤差、体積ひずみ det F − 1 の推定・真値・誤差。"
      "推定と真値は同じ色のスケール、誤差は行ごとのスケール。推定が無効な画素は表示しない。場は様式化した合成で、実験で較正していない。）\n")

    a("## 7. 処理時間\n")
    a("| 段階 | 時間 |")
    a("|---|---:|")
    names = {"generate": "動画と真値の生成", "track": "追跡(CLI 全体、動画の読み込みと .npy の書き出しを含む)",
             "evaluate": "真値との比較", "figures": "overview.png と curves.png", "animate": "strain_animation.mp4"}
    for k, v in names.items():
        if k in st:
            a(f"| {v} | {_fmt_s(st[k])} |")
    a(f"| **合計** | **{_fmt_s(st['total'])}** |")
    a("")
    pf = cli["wall_seconds"] / max(n - 1, 1)
    a(f"- 追跡のアルゴリズム時間(各フレームの coarse + fine + strain の合計): coarse {algo['t_coarse']:.1f} s、fine {algo['t_fine']:.1f} s、"
      f"strain {algo['t_strain']:.1f} s。CLI の壁時計 {cli['wall_seconds']:.1f} s(1 フレームあたり {pf:.2f} s)、"
      f"最大メモリ {_mem(cli.get('peak_footprint_bytes') or cli.get('max_rss_bytes'))}。")
    la = metrics["env"]["loadavg"]
    a(f"- 実行時のマシンの負荷平均(1/5/15 分): {la[0]:.1f} / {la[1]:.1f} / {la[2]:.1f}。他のジョブが走っていると時間は大きく伸びる。")
    a("")

    a("## 8. 誤差が大きい場所とその理由\n")
    X = ev.extra
    a(f"- **せん断帯の中**: 帯の幅(ひずみの半値全幅で約 {fwhm:.0f} px)が、追跡のフィルタの窓(σ_w = {cfg['sigma_w']:g} px、"
      f"波長 {cfg['lam'] * cfg['scales'][-1]:.0f} px までのフィルタ)とひずみの局所 2 次近似の窓({cfg['strain_win']} px)と同程度なので、"
      f"帯の変位の食い違いが平滑化され、ひずみの山は低く広くなる(有効画素での gmax の 99.9 パーセンタイルは真値の {peak_ratio * 100:.0f} %、"
      f"ガード外の全画素の真値に対しては {peak_ratio_all * 100:.0f} %)。"
      f"最終フレームでは、帯の画素はガード外の有効画素の {X['band_pixel_share'] * 100:.1f} % しかないのに、変位の二乗誤差の "
      f"{X['band_sq_error_share'] * 100:.0f} % を占める(帯の中の変位誤差は外の {band_ratio:.1f} 倍)。")
    a(f"- **沈下が進むほど誤差が増える**: 帯のすべり量は s_m に比例して増え(s_m ∝ s³ から s へ)、帯のひずみも増える(有効画素での真値の gmax の "
      f"99.9 パーセンタイル {first['gmax_peak_true']:.3f} → {last['gmax_peak_true']:.3f}、ガード外の全画素では {first['gmax_peak_true_all']:.3f} → "
      f"{last['gmax_peak_true_all']:.3f})。変位誤差の RMS は {first['disp_rms']:.4f} px(フレーム 1)から {last['disp_rms']:.4f} px(最終)へ増える。"
      f"帯の外の誤差は {first['disp_rms_out']:.4f} px(ノイズによる下限に近い)から {last['disp_rms_out']:.4f} px へ増える。"
      f"帯の外にも、帯の応答が窓の幅ぶん広がる分(帯の外の画素には、帯の判定のしきい値 = 最大の {int(100 * 0.4)} % ≈ "
      f"{0.4 * fsum['gmax_p999']:.2f} の直下にある帯の裾が含まれる)と、局所 2 次近似の窓の中の曲率による系統誤差が加わる。"
      f"拡散した破壊前の部分だけの gmax は、基礎の下の中央で最大 {fsum['gmax_pre_middle_max']:.2f}、縁の帯でも最大 {fsum['gmax_pre_max']:.2f} "
      f"にすぎず、帯の外の誤差の主な原因ではない。")
    de, ge = X["disp_error_max_at"], X["gmax_error_max_at"]
    a(f"- **楔の頂点と、フーチングの縁の近く**: 左右のすべり線が交わる頂点付近と、扇形の中心でありすべり線が地表に出るフーチングの縁の近くは、"
      f"帯が交わり曲率も大きい(この 2 か所は特異点で、真値も正則化してある)。最終フレームで変位誤差が最大の画素は (x', y') = ({de['x']:.0f}, {de['y']:.0f}) px で {de['value']:.2f} px、"
      f"gmax の誤差が最大の画素は ({ge['x']:.0f}, {ge['y']:.0f}) px で {ge['value']:.2f}"
      f"(x' はフーチングの中心線から右、y' は初期の地表から下。楔の頂点は y' = {p.ha:.0f} px、フーチングの縁は x' = ±{p.b:g} px)。"
      f"フーチングの中心から {2.5 * p.b:.0f} px 以内の画素はガード外の {X['near_footing_pixel_share'] * 100:.0f} % で、"
      f"二乗誤差の {X['near_footing_sq_error_share'] * 100:.0f} % がそこにある。")
    k_lost = next((r["k"] for r in rows if r["rigid_valid_frac"] < 0.9), None)
    rvf0, rvf1 = rows[0]["rigid_valid_frac"], rows[-1]["rigid_valid_frac"]
    ok_ref = np.isfinite(S["settlement_est_last_valid"])
    a(f"- **楔の内側が終盤に無効になる**: 強いせん断帯(最終フレームの真値 gmax は帯で最大 {fsum['gmax_max']:.1f} 近く)では追跡の信頼度が下がり、"
      f"その低信頼の領域とガード距離({cfg['edge_guard_px']:.0f} px)の帯が、帯に挟まれた楔の内側(幅は深さとともに狭くなる)を覆う。"
      f"沈下の推定に使う楔の画素のうち有効な画素の割合は、フレーム 1 で {rvf0:.2f}、最終フレームで {rvf1:.2f}"
      + (f"(フレーム {k_lost} 以降 0.9 を下回る)" if k_lost else "") +
      (f"。残った画素の中央値は真値に合っている({S['settlement_est_last_valid']:.2f} px 対 {S['settlement_ref_last_valid']:.2f} px、フレーム {S['settlement_last_valid_frame']})が、"
       if ok_ref else "。") +
      "頂点付近の最大のひずみは追跡できない。"
      "このため curves.png の「真値の 99.9 パーセンタイル(有効画素)」は、最大のひずみの画素が無効になる所で下がる(点線は全画素)。"
      + ("小さい版(`--size small`)では楔が 60 px ほどと狭く、設定の沈下が大きいと終盤に楔が全部無効になる。" if args.size == "small" else ""))
    a(f"- **ROI の縁**: ROI の外側(壁、フーチング、空気)は追跡できないので、縁から {cfg['edge_guard_px']:.0f} px 以内は設計上無効にしてある"
      f"(ROI 全体に対する有効画素の割合の上限は約 {S['interior_pixels'] / S['roi_pixels']:.2f})。フーチング直下の地表近くも ROI の上縁の内側のガード帯で無効になる。")
    a("- **ひずみは変位より誤差が目立つ**: ひずみは変位の空間微分(局所 2 次近似)なので、変位の誤差とノイズが微分で増幅される。"
      "体積ひずみ(det F − 1)は帯の脇で符号の違う山をつくる真値を、平滑化のため弱めて出す。")
    a("")

    a("## 9. 再実行\n")
    a("```sh")
    a("cd python && python -m demo.run_demo                # 既定: 全サイズ、60 フレーム、Rust、demo_out/ に出力")
    a("python python/demo/run_demo.py --out /tmp/demo       # リポジトリのルートから、出力先を指定")
    a("python -m demo.run_demo --size small                 # 小さい版(数十秒、スモークテスト用。精度は全サイズより低い)")
    a("python -m demo.run_demo --impl python                # 遅い Python 参照実装で追跡(Rust との比較用)")
    a("python -m demo.run_demo --no-animation               # 動画の生成を省く")
    a("```\n")
    a("Rust の実行ファイルが無いときは `cargo build --release -p optflow-cli`(`rust/` で)を自動で実行する。")
    a("")

    a("## 10. 出力ファイル\n")
    a("| ファイル | 内容 |")
    a("|---|---|")
    a("| `footing.mp4` | 合成した入力動画(H.264) |")
    a("| `roi.json` | ROI の多角形(フレーム 0 の画素座標) |")
    a("| `truth.npz` | 真値: フレームごとの沈下量、せん断帯などの形状関数(任意のフレームの真値を作れる)、選んだフレームの真値の場 |")
    a("| `track/` | トラッカーの設定 (`config.json`)、クロップ (`crop.json`)。`--keep-fields` のときは各フレームの `.npy` も |")
    a("| `overview.png`, `curves.png` | 図 |")
    a("| `strain_animation.mp4` | 入力動画と gmax の重ね合わせ(現在位置) |")
    a("| `per_frame.csv`, `metrics.json` | フレームごとの誤差と、実行条件・時間・要約 |")
    a("| `cli_stdout.txt`, `cli_stderr.txt` | トラッカーの出力(警告はここ) |")
    a("")
    return "\n".join(L) + "\n"

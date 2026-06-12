"""
ADCE Missing Neighbor Detector — Streamlit App
================================================
Usage: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import math
import folium
from streamlit_folium import st_folium
import io

st.set_page_config(
    page_title="ADCE Missing Neighbor Detector",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════
EARTH_R = 6371.0

def haversine(lat1, lon1, lat2, lon2):
    dlat, dlon = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return EARTH_R * 2 * math.asin(math.sqrt(a))

def az_offset(lat, lon, azimuth, dist_deg):
    rad = math.radians(azimuth)
    return lat + dist_deg*math.cos(rad), lon + dist_deg*math.sin(rad)/math.cos(math.radians(lat))

def draw_sector(m_layer, lat, lon, azimuth, beamwidth, radius_deg, color, fill_opacity, popup_text):
    bw = min(beamwidth, 120)
    points = [(lat, lon)]
    for angle in np.arange(azimuth-bw/2, azimuth+bw/2+1, 2):
        rad = math.radians(angle)
        points.append((lat + radius_deg*math.cos(rad),
                       lon + radius_deg*math.sin(rad)/math.cos(math.radians(lat))))
    points.append((lat, lon))
    folium.Polygon(locations=points, color=color, weight=1.5,
                   fill=True, fill_color=color, fill_opacity=fill_opacity,
                   popup=folium.Popup(popup_text, max_width=350)).add_to(m_layer)

@st.cache_data
def load_gcell(file):
    df = pd.read_csv(file, sep='\t', encoding='latin1')
    df.columns = df.columns.str.strip()
    df['LAC_CI'] = df['LAC'].astype(str).str.strip() + '_' + df['CI'].astype(str).str.strip()
    for col in ['Longitude','Latitude','azimuth','beamwidth']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Longitude','Latitude','azimuth','beamwidth'])
    df['Kabupaten'] = df['Kabupaten'].str.strip().str.upper()
    df['Site_Type'] = df['Site_Type'].str.strip()
    df['BSC'] = df['BSC'].str.strip()
    return df

@st.cache_data
def load_adce(file):
    df = pd.read_csv(file, sep='\t', encoding='latin1')
    df.columns = df.columns.str.strip()
    df['LAC_CI_Source'] = df['LAC_CI_Source'].astype(str).str.strip()
    df['LAC_CI_Target'] = df['LAC_CI_Target'].astype(str).str.strip()
    return df

@st.cache_data
def detect_missing(gcell, adce_set_list, urban_km, suburban_km, rural_km, suburban_list,
                   filter_bsc, filter_kab, filter_site, exclude_ibc):
    adce_set = set(adce_set_list)
    work = gcell.copy()
    if exclude_ibc:
        work = work[work['Site_Type'] == 'MACRO']

    # Classify
    sub_upper = [s.upper() for s in suburban_list]
    def get_threshold(kab):
        if kab.startswith('KOTA'):
            return urban_km, 'Urban'
        elif kab in sub_upper:
            return suburban_km, 'Suburban'
        return rural_km, 'Rural'

    work[['threshold_km','area_type']] = work['Kabupaten'].apply(
        lambda k: pd.Series(get_threshold(k)))

    # Apply filters
    if filter_bsc:
        work = work[work['BSC'] == filter_bsc]
    if filter_kab:
        work = work[work['Kabupaten'] == filter_kab.upper()]
    if filter_site:
        work = work[work['SiteID'] == filter_site]

    if len(work) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # All cells for target
    all_cells = gcell[['LAC_CI','Cellname','SiteID','Latitude','Longitude',
                       'azimuth','beamwidth','Kabupaten','BSC']].reset_index(drop=True)
    if exclude_ibc:
        all_cells = gcell[gcell['Site_Type']=='MACRO'][
            ['LAC_CI','Cellname','SiteID','Latitude','Longitude',
             'azimuth','beamwidth','Kabupaten','BSC']].reset_index(drop=True)

    coords_rad = np.radians(all_cells[['Latitude','Longitude']].values)
    tree = BallTree(coords_rad, metric='haversine')

    src_cells = work[['LAC_CI','Cellname','SiteID','Latitude','Longitude',
                      'azimuth','beamwidth','Kabupaten','BSC','threshold_km','area_type']].reset_index(drop=True)

    max_threshold = work['threshold_km'].max()
    src_coords_rad = np.radians(src_cells[['Latitude','Longitude']].values)
    indices_list = tree.query_radius(src_coords_rad, r=max_threshold/EARTH_R)

    # Site → threshold lookup
    site_threshold = work.drop_duplicates('SiteID').set_index('SiteID')[['threshold_km','area_type']].to_dict('index')

    missing = []
    for i, neighbors in enumerate(indices_list):
        src = src_cells.iloc[i]
        st_info = site_threshold.get(src['SiteID'], {'threshold_km': rural_km, 'area_type': 'Rural'})
        threshold = st_info['threshold_km']
        area_type = st_info['area_type']

        for j in neighbors:
            tgt = all_cells.iloc[j]
            if src['SiteID'] == tgt['SiteID']:
                continue
            dist = haversine(src['Latitude'], src['Longitude'], tgt['Latitude'], tgt['Longitude'])
            if dist > threshold:
                continue
            if (src['LAC_CI'], tgt['LAC_CI']) in adce_set:
                continue

            missing.append({
                'Source_Cell': src['Cellname'],
                'Source_SiteID': src['SiteID'],
                'Source_LAC_CI': src['LAC_CI'],
                'Source_Azimuth': int(src['azimuth']),
                'Source_BSC': src['BSC'],
                'Target_Cell': tgt['Cellname'],
                'Target_SiteID': tgt['SiteID'],
                'Target_LAC_CI': tgt['LAC_CI'],
                'Target_Azimuth': int(tgt['azimuth']),
                'Distance_km': round(dist, 2),
                'Kabupaten': src['Kabupaten'],
                'Area_Type': area_type,
                'Threshold_km': threshold,
            })

    df_missing = pd.DataFrame(missing)
    if len(df_missing) == 0:
        return df_missing, pd.DataFrame()

    # Site summary
    lacci_to_site = gcell.set_index('LAC_CI')['SiteID'].to_dict()
    adce_df = pd.DataFrame(list(adce_set), columns=['src','tgt'])
    adce_df['src_site'] = adce_df['src'].map(lacci_to_site)
    adce_df['tgt_site'] = adce_df['tgt'].map(lacci_to_site)
    existing_counts = adce_df.dropna().groupby(['src_site','tgt_site']).size().reset_index(name='existing_count')

    site_summary = df_missing.groupby(['Source_SiteID','Target_SiteID']).agg(
        missing_count=('Source_Cell','size'),
        min_dist=('Distance_km','min'),
        area_type=('Area_Type','first'),
        threshold_km=('Threshold_km','first'),
        kabupaten=('Kabupaten','first'),
        source_bsc=('Source_BSC','first'),
    ).reset_index()

    site_summary = site_summary.merge(
        existing_counts,
        left_on=['Source_SiteID','Target_SiteID'],
        right_on=['src_site','tgt_site'],
        how='left'
    ).drop(columns=['src_site','tgt_site'], errors='ignore')
    site_summary['existing_count'] = site_summary['existing_count'].fillna(0).astype(int)
    site_summary['priority'] = site_summary['existing_count'].apply(
        lambda x: 'CRITICAL' if x == 0 else 'PARTIAL')

    return df_missing.sort_values(['Source_BSC','Source_SiteID','Distance_km']), \
           site_summary.sort_values(['priority','min_dist'])


def build_map(gcell, adce_set, df_missing, focus_site, radius_km):
    focus_cells = gcell[gcell['SiteID'] == focus_site]
    if len(focus_cells) == 0:
        return None
    center_lat = focus_cells['Latitude'].mean()
    center_lon = focus_cells['Longitude'].mean()

    sites = gcell.drop_duplicates('SiteID')[['SiteID','Latitude','Longitude']].copy()
    sites['dist'] = sites.apply(lambda r: haversine(center_lat, center_lon, r['Latitude'], r['Longitude']), axis=1)
    nearby_sites = set(sites[sites['dist'] <= radius_km]['SiteID']) | {focus_site}
    nearby_cells = gcell[gcell['SiteID'].isin(nearby_sites)]
    focus_lacci = set(focus_cells['LAC_CI'])

    cell_info = {}
    for _, c in nearby_cells.iterrows():
        cell_info[c['LAC_CI']] = {
            'lat': c['Latitude'], 'lon': c['Longitude'],
            'az': c['azimuth'], 'bw': c['beamwidth'],
            'name': c['Cellname'], 'site': c['SiteID']
        }

    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles='CartoDB positron')
    lg_sectors  = folium.FeatureGroup(name='📡 Cell sectors', show=True)
    lg_existing = folium.FeatureGroup(name='🟢 Existing ADCE', show=True)
    lg_missing  = folium.FeatureGroup(name='🔴 Missing ADCE', show=True)
    lg_labels   = folium.FeatureGroup(name='🏷️ Site labels', show=True)

    BEAM_LEN = 0.0012
    LINE_OFFSET = 0.0006

    for _, cell in nearby_cells.iterrows():
        is_focus = cell['SiteID'] == focus_site
        color = '#e74c3c' if is_focus else '#27ae60'
        popup = (f"<b>{cell['Cellname']}</b><br>"
                 f"LAC_CI: {cell['LAC_CI']}<br>"
                 f"Az: {int(cell['azimuth'])}° | BW: {int(cell['beamwidth'])}°")
        draw_sector(lg_sectors, cell['Latitude'], cell['Longitude'],
                    cell['azimuth'], cell['beamwidth'],
                    BEAM_LEN, color, 0.7 if is_focus else 0.45, popup)

    for sid in nearby_sites:
        sc = nearby_cells[nearby_cells['SiteID']==sid]
        slat, slon = sc['Latitude'].mean(), sc['Longitude'].mean()
        is_focus = sid == focus_site
        dist = haversine(center_lat, center_lon, slat, slon)
        label = sid if is_focus else f'{sid} ({dist:.1f}km)'
        folium.Marker(
            location=[slat, slon],
            icon=folium.DivIcon(
                html=f'<div style="font-size:{"11" if is_focus else "9"}px;'
                     f'font-weight:{"bold" if is_focus else "normal"};'
                     f'color:{"#c0392b" if is_focus else "#2c3e50"};white-space:nowrap;'
                     f'text-shadow:1px 1px 1px #fff,-1px -1px 1px #fff,'
                     f'1px -1px 1px #fff,-1px 1px 1px #fff">{label}</div>',
                icon_size=(0,0), icon_anchor=(0,-18))
        ).add_to(lg_labels)

    # Existing ADCE lines
    existing_drawn = set()
    for src_lacci in focus_lacci:
        src = cell_info.get(src_lacci)
        if not src: continue
        for (s, t) in adce_set:
            if s != src_lacci or t not in cell_info: continue
            if (s, t) in existing_drawn: continue
            existing_drawn.add((s, t))
            tgt = cell_info[t]
            src_tip = az_offset(src['lat'], src['lon'], src['az'], LINE_OFFSET)
            tgt_tip = az_offset(tgt['lat'], tgt['lon'], tgt['az'], LINE_OFFSET)
            dist = haversine(src['lat'], src['lon'], tgt['lat'], tgt['lon'])
            popup = (f"✅ <b>EXISTING</b><br><b>Src:</b> {src['name']}<br>"
                     f"<b>Tgt:</b> {tgt['name']}<br>Dist: {dist:.2f} km")
            folium.PolyLine(locations=[src_tip, tgt_tip],
                            color='#2ecc71', weight=2.5, opacity=0.85, dash_array='8,4',
                            popup=folium.Popup(popup, max_width=300)).add_to(lg_existing)

    # Missing ADCE lines
    missing_drawn = set()
    for src_lacci in focus_lacci:
        src = cell_info.get(src_lacci)
        if not src: continue
        for tgt_lacci, tgt in cell_info.items():
            if tgt['site'] == focus_site: continue
            if (src_lacci, tgt_lacci) in adce_set: continue
            dist = haversine(src['lat'], src['lon'], tgt['lat'], tgt['lon'])
            if dist > radius_km: continue
            key = (src_lacci, tgt_lacci)
            if key in missing_drawn: continue
            missing_drawn.add(key)
            src_tip = az_offset(src['lat'], src['lon'], src['az'], LINE_OFFSET)
            tgt_tip = az_offset(tgt['lat'], tgt['lon'], tgt['az'], LINE_OFFSET)
            popup = (f"❌ <b>MISSING</b><br><b>Src:</b> {src['name']}<br>"
                     f"<b>Tgt:</b> {tgt['name']}<br>Dist: {dist:.2f} km")
            folium.PolyLine(locations=[src_tip, tgt_tip],
                            color='#e74c3c', weight=2, opacity=0.6, dash_array='4,6',
                            popup=folium.Popup(popup, max_width=300)).add_to(lg_missing)

    lg_sectors.add_to(m); lg_existing.add_to(m); lg_missing.add_to(m); lg_labels.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    legend = f"""<div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
         padding:14px 18px;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,0.25);
         font-size:12px;line-height:2">
      <b style="font-size:14px">🗺️ {focus_site}</b><br>
      <span style="color:#e74c3c">■</span> Focus site &nbsp;
      <span style="color:#27ae60">■</span> Neighbors<br>
      <span style="color:#2ecc71">━━</span> Existing ({len(existing_drawn)}) &nbsp;
      <span style="color:#e74c3c">╌╌</span> Missing ({len(missing_drawn)})
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))

    return m

# ═══════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════
with st.sidebar:
    st.title("📡 ADCE Detector")
    st.caption("Missing Neighbor Detection for 2G GSM")

    st.subheader("📂 Upload files")
    gcell_file = st.file_uploader("GCELL (cell reference)", type=['txt','csv'], key='gcell')
    adce_file = st.file_uploader("ADCE (neighbor list)", type=['txt','csv'], key='adce')

    st.divider()
    st.subheader("⚙️ Threshold settings")

    threshold_mode = st.radio("Mode", ['Fixed','ISD (auto)'], horizontal=True)

    if threshold_mode == 'Fixed':
        col1, col2, col3 = st.columns(3)
        urban_km = col1.number_input("Urban", 0.1, 10.0, 1.0, 0.1, help="KOTA*")
        suburban_km = col2.number_input("Suburban", 0.1, 10.0, 2.0, 0.1)
        rural_km = col3.number_input("Rural", 0.1, 15.0, 5.0, 0.5)

        suburban_input = st.text_input(
            "Suburban kabupaten (comma separated)",
            "DELI SERDANG, KARO, SIMALUNGUN, LANGKAT, SERDANG BEDAGAI",
            help="Kabupaten yang diklasifikasi suburban"
        )
        suburban_list = [s.strip().upper() for s in suburban_input.split(',') if s.strip()]
    else:
        col1, col2 = st.columns(2)
        isd_mult = col1.number_input("ISD multiplier", 1.0, 5.0, 2.0, 0.5)
        isd_neighbors = col2.number_input("Nearest N", 1, 10, 3, 1)
        col3, col4 = st.columns(2)
        isd_min = col3.number_input("Min km", 0.1, 5.0, 0.5, 0.1)
        isd_max = col4.number_input("Max km", 1.0, 15.0, 5.0, 0.5)
        # Placeholder — use fixed for now
        urban_km, suburban_km, rural_km = 1.0, 2.0, 5.0
        suburban_list = []

    st.divider()
    st.subheader("🔍 Filters")
    exclude_ibc = st.checkbox("Exclude IBC (indoor)", value=True)

# ═══════════════════════════════════════════
#  MAIN AREA
# ═══════════════════════════════════════════
if not gcell_file or not adce_file:
    st.markdown("## 📡 ADCE Missing Neighbor Detector")
    st.info("Upload **GCELL** dan **ADCE** file di sidebar untuk memulai.")

    with st.expander("ℹ️ Cara penggunaan"):
        st.markdown("""
        **Step 1:** Upload file GCELL (cell reference) dan ADCE (neighbor list) di sidebar

        **Step 2:** Set threshold jarak per area type (Urban / Suburban / Rural)

        **Step 3:** Pilih tab:
        - **Detect All** — jalankan deteksi untuk semua site, download CSV
        - **Site Inspector** — pilih site ID, lihat map interaktif
        - **Summary** — ringkasan statistik
        """)
    st.stop()

# Load data
gcell = load_gcell(gcell_file)
adce = load_adce(adce_file)
adce_set = set(zip(adce['LAC_CI_Source'], adce['LAC_CI_Target']))
adce_set_list = list(adce_set)

# Sidebar filters (need data loaded first)
with st.sidebar:
    bsc_list = [''] + sorted(gcell['BSC'].unique().tolist())
    filter_bsc = st.selectbox("Filter BSC", bsc_list, help="Kosongkan = semua")

    kab_list = [''] + sorted(gcell['Kabupaten'].unique().tolist())
    filter_kab = st.selectbox("Filter Kabupaten", kab_list)

    filter_site = st.text_input("Filter Site ID", '', placeholder="e.g. 02MDN0268")

# Data stats in sidebar
with st.sidebar:
    st.divider()
    st.caption(f"📊 Loaded: {len(gcell):,} cells | {gcell['SiteID'].nunique():,} sites")
    st.caption(f"🔗 ADCE: {len(adce_set):,} relations")

# ═══════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔎 Detect all", "🗺️ Site inspector", "📊 Summary"])

# ── TAB 1: DETECT ALL ──
with tab1:
    st.subheader("Detect missing ADCE — all sites")

    if st.button("▶ Run detection", type="primary", use_container_width=True):
        with st.spinner("Detecting missing neighbors..."):
            df_missing, site_summary = detect_missing(
                gcell, adce_set_list,
                urban_km, suburban_km, rural_km, suburban_list,
                filter_bsc, filter_kab, filter_site, exclude_ibc
            )
            st.session_state['df_missing'] = df_missing
            st.session_state['site_summary'] = site_summary

    if 'df_missing' in st.session_state and len(st.session_state['df_missing']) > 0:
        df_missing = st.session_state['df_missing']
        site_summary = st.session_state['site_summary']

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total missing", f"{len(df_missing):,}")
        col2.metric("Site pairs", f"{len(site_summary):,}")
        col3.metric("Critical", f"{(site_summary['priority']=='CRITICAL').sum():,}")
        col4.metric("Partial", f"{(site_summary['priority']=='PARTIAL').sum():,}")

        # Area breakdown
        st.markdown("**By area type:**")
        area_stats = df_missing.groupby('Area_Type').agg(
            count=('Source_Cell','size'),
            avg_dist=('Distance_km','mean'),
            threshold=('Threshold_km','first')
        ).reset_index()
        st.dataframe(area_stats, use_container_width=True, hide_index=True)

        # Downloads
        st.markdown("---")
        col_a, col_b = st.columns(2)

        csv_cell = df_missing.to_csv(index=False).encode('utf-8')
        col_a.download_button(
            "📥 Download cell detail CSV",
            csv_cell,
            "missing_adce_cell_detail.csv",
            "text/csv",
            use_container_width=True
        )

        csv_site = site_summary.to_csv(index=False).encode('utf-8')
        col_b.download_button(
            "📥 Download site summary CSV",
            csv_site,
            "missing_adce_site_summary.csv",
            "text/csv",
            use_container_width=True
        )

        # Preview
        with st.expander(f"Preview cell detail ({len(df_missing):,} rows)"):
            st.dataframe(df_missing.head(100), use_container_width=True, hide_index=True)

        with st.expander(f"Preview site summary ({len(site_summary):,} rows)"):
            st.dataframe(site_summary.head(100), use_container_width=True, hide_index=True)

    elif 'df_missing' in st.session_state:
        st.warning("No missing ADCE found with current filters.")

# ── TAB 2: SITE INSPECTOR ──
with tab2:
    st.subheader("Site inspector — interactive map")

    col_site, col_radius = st.columns([3, 1])

    site_list = sorted(gcell['SiteID'].unique().tolist())
    default_idx = site_list.index('02MDN0337') if '02MDN0337' in site_list else 0

    selected_site = col_site.selectbox("Select Site ID", site_list, index=default_idx)
    map_radius = col_radius.number_input("Radius (km)", 0.5, 10.0, 3.0, 0.5)

    if selected_site:
        # Site info
        site_cells = gcell[gcell['SiteID'] == selected_site]
        site_kab = site_cells['Kabupaten'].iloc[0] if len(site_cells) > 0 else '-'
        site_bsc = site_cells['BSC'].iloc[0] if len(site_cells) > 0 else '-'

        col_i1, col_i2, col_i3 = st.columns(3)
        col_i1.markdown(f"**Kabupaten:** {site_kab}")
        col_i2.markdown(f"**BSC:** {site_bsc}")
        col_i3.markdown(f"**Cells:** {len(site_cells)}")

        with st.spinner("Generating map..."):
            fmap = build_map(gcell, adce_set, None, selected_site, map_radius)

        if fmap:
            st_folium(fmap, use_container_width=True, height=550)

        # Site detail table
        if 'df_missing' in st.session_state and len(st.session_state['df_missing']) > 0:
            site_missing = st.session_state['df_missing'][
                st.session_state['df_missing']['Source_SiteID'] == selected_site]

            if len(site_missing) > 0:
                st.markdown(f"**Missing neighbors from {selected_site}: {len(site_missing)}**")

                by_target = site_missing.groupby('Target_SiteID').agg(
                    missing=('Source_Cell','size'),
                    min_dist=('Distance_km','min')
                ).sort_values('min_dist').reset_index()
                st.dataframe(by_target, use_container_width=True, hide_index=True)
            else:
                st.success(f"No missing ADCE for {selected_site}")
        else:
            st.info("Run detection di tab 'Detect all' dulu untuk melihat detail missing per site.")

# ── TAB 3: SUMMARY ──
with tab3:
    st.subheader("Summary dashboard")

    if 'df_missing' not in st.session_state:
        st.info("Run detection di tab 'Detect all' dulu.")
        st.stop()

    df_missing = st.session_state['df_missing']
    site_summary = st.session_state['site_summary']

    if len(df_missing) == 0:
        st.warning("No data.")
        st.stop()

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total missing", f"{len(df_missing):,}")
    col2.metric("Affected cells", f"{df_missing['Source_Cell'].nunique():,}")
    col3.metric("Affected sites", f"{df_missing['Source_SiteID'].nunique():,}")
    col4.metric("Avg distance", f"{df_missing['Distance_km'].mean():.2f} km")

    st.markdown("---")

    # BSC breakdown
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Top 10 BSC by missing count**")
        bsc_stats = df_missing.groupby('Source_BSC').size().reset_index(name='missing_count')
        bsc_stats = bsc_stats.sort_values('missing_count', ascending=False).head(10)
        st.bar_chart(bsc_stats.set_index('Source_BSC'))

    with col_r:
        st.markdown("**By area type**")
        area_pie = df_missing['Area_Type'].value_counts().reset_index()
        area_pie.columns = ['Area_Type','count']
        st.bar_chart(area_pie.set_index('Area_Type'))

    # Distance distribution
    st.markdown("**Distance distribution (km)**")
    dist_hist = pd.cut(df_missing['Distance_km'],
                       bins=[0,0.5,1,1.5,2,2.5,3,4,5],
                       labels=['0-0.5','0.5-1','1-1.5','1.5-2','2-2.5','2.5-3','3-4','4-5'])
    dist_counts = dist_hist.value_counts().sort_index().reset_index()
    dist_counts.columns = ['range_km','count']
    st.bar_chart(dist_counts.set_index('range_km'))

    # Priority breakdown
    st.markdown("**Site-pair priority**")
    prio = site_summary['priority'].value_counts().reset_index()
    prio.columns = ['Priority','Count']
    st.dataframe(prio, use_container_width=True, hide_index=True)

    # Top 20 closest missing
    st.markdown("**Top 20 closest missing site pairs**")
    st.dataframe(
        site_summary.head(20)[['Source_SiteID','Target_SiteID','min_dist',
                                'missing_count','existing_count','priority','kabupaten']],
        use_container_width=True, hide_index=True
    )

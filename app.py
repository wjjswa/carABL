"""Read real experiment artifacts locally or via an SSH tunnel; no fabricated scores."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title='ABL vehicle damage experiments', layout='wide')
st.title('Vehicle damage · experiment results')
root = Path(st.sidebar.text_input('Results directory', 'outputs')).expanduser()
image_root = Path(st.sidebar.text_input('Dataset image root', 'data/kaggle')).expanduser()
st.caption('Experiments 1–4 and ActiveGCD 5B. Scores appear only after running the corresponding experiments.')
tabs = st.tabs(['Comparison', 'Image predictions', 'Category discovery', 'Run details'])

with tabs[0]:
    direct_metrics = root / 'exp1' / 'metrics.json'
    if direct_metrics.exists():
        st.subheader('Experiment 1 · direct transfer')
        st.json(json.loads(direct_metrics.read_text()))
    summary = root / 'summary.csv'
    if summary.exists():
        df = pd.read_csv(summary)
        if len(df):
            ev = st.selectbox('Evaluation', sorted(df.evaluation.unique()))
            part = df[df.evaluation == ev].copy()
            part['series'] = 'Exp ' + part.experiment.astype(str) + ' · ' + part.method
            pivot = part.pivot_table(index='fraction', columns='series', values='macro_f1', aggfunc='mean')
            st.line_chart(pivot)
            st.dataframe(part, width='stretch')
            st.download_button('Download results CSV', df.to_csv(index=False), 'results.csv')
        else:
            st.info('No completed classification results yet.')
    else:
        st.info('Run `carabl export --output outputs` after evaluation to build the comparison table.')
    for plot in sorted(root.glob('*label_efficiency.png')):
        st.image(str(plot), caption=plot.stem)

with tabs[1]:
    prediction_files = sorted(root.rglob('*predictions.csv'))
    if prediction_files:
        filename = st.selectbox('Prediction file', prediction_files, format_func=lambda p: str(p.relative_to(root)))
        data = pd.read_csv(filename)
        if 'correct' in data and st.checkbox('Show errors only'):
            data = data[~data.correct.astype(bool)]
        if len(data):
            idx = st.slider('Image', 0, len(data)-1, 0) if len(data) > 1 else 0
            row = data.iloc[idx]
            cols = st.columns(3)
            path = image_root / row.path
            if path.exists():
                cols[0].image(str(path), caption=f'Label: {row.label}')
            case = root / 'exp1' / 'cases' / str(row.id)
            for col, name in zip(cols[1:], ['boxes_overlay.jpg', 'masks_overlay.jpg']):
                if (case / name).exists():
                    col.image(str(case / name), caption=name.replace('_', ' '))
            st.write({'prediction': row.get('prediction'), 'model score': row.get('confidence')})
            st.caption('Boxes/masks in Experiment 1 are predictions; localisation accuracy is not measured without location annotations.')
            bank_file = filename.parent / 'knn_reference_bank.csv'
            feature_file = filename.parent / 'knn_features.npz'
            if filename.name.startswith('knn_') and bank_file.exists() and feature_file.exists():
                import numpy as np
                bank = pd.read_csv(bank_file)
                original = pd.read_csv(filename)
                pos = original.index[original.id == row.id][0]
                with np.load(feature_file) as arrays:
                    nearest = arrays['neighbours'][pos][:5]
                st.write('Nearest labelled training examples')
                columns = st.columns(len(nearest))
                for col, n in zip(columns, nearest):
                    neighbour = bank.iloc[n]
                    p = image_root / neighbour.path
                    if p.exists():
                        col.image(str(p), caption=neighbour.label)
            st.dataframe(data, width='stretch')
        else:
            st.info('No images match this filter.')
    else:
        st.info('No saved predictions yet.')

with tabs[2]:
    rounds = sorted(root.glob('exp5b/*/round_summary.csv'))
    if rounds:
        for filename in rounds:
            st.subheader(filename.parent.name)
            df = pd.read_csv(filename)
            st.line_chart(df.set_index('annotations')[['old_accuracy', 'novel_semantic_accuracy', 'offline_cluster_accuracy']])
            st.dataframe(df)
        queries = sorted(root.glob('exp5b/*/round*/queried_annotations.csv'))
        if queries:
            q = st.selectbox('Annotation round', queries, format_func=lambda p: str(p.relative_to(root)))
            annotated = pd.read_csv(q)
            st.dataframe(annotated[['id', 'path', 'label']])
            for _, row in annotated.head(12).iterrows():
                p = image_root / row.path
                if p.exists():
                    st.image(str(p), caption=f'Queried label: {row.label}', width=260)
        st.caption('Offline cluster accuracy permits an evaluation-only label permutation. Semantic accuracy uses only labels revealed in training/annotation.')
    else:
        st.info('ActiveGCD results will appear after its base training and annotation rounds.')

with tabs[3]:
    configs = sorted(root.rglob('provenance.json'))
    if configs:
        p = st.selectbox('Recorded run', configs, format_func=lambda p: str(p.parent.relative_to(root)))
        st.json(json.loads(p.read_text()))
    else:
        st.info('No run metadata yet.')

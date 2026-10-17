"""Verifica los totales de la evaluación sin volver a ejecutar el modelo."""
import csv,json,math,hashlib
from collections import Counter
from pathlib import Path

here=Path(__file__).resolve().parent
manifest=json.loads((here/'division/manifiesto.json').read_text(encoding='utf-8'))
records=manifest['registros']
train=[r for r in records if r['conjunto']=='entrenamiento']
test=[r for r in records if r['conjunto']=='prueba']
assert len(test)==150
assert set(Counter(r['clase'] for r in test).values())=={30}
for key in ['grupo','pixeles_sha256','sha256']:
    assert not ({r[key] for r in train}&{r[key] for r in test}),key
runpath=sorted((here/'resultados').glob('*/resumen.json'))[-1]
summary=json.loads(runpath.read_text(encoding='utf-8'))
assert summary['manifiesto_sha256']==hashlib.sha256((here/'division/manifiesto.json').read_bytes()).hexdigest()
for subset in ['entrenamiento','prueba']:
    metrics=summary['resultados'][subset]
    with (runpath.parent/f'predicciones_{subset}.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==metrics['cantidad']
    labels=metrics['etiquetas'];index={l:i for i,l in enumerate(labels)}
    matrix=[[0]*5 for _ in labels]
    for r in rows:matrix[index[r['real']]][index[r['predicha']]]+=1
    assert matrix==metrics['matriz_confusion']
    assert math.isclose(sum(r['real']==r['predicha'] for r in rows)/len(rows),metrics['exactitud'])
    accepted=[r for r in rows if r['aceptada_app']=='True']
    assert len(accepted)==metrics['aceptadas_app']
    if accepted:assert math.isclose(sum(r['real']==r['predicha'] for r in accepted)/len(accepted),metrics['exactitud_aceptadas_app'])
    print(subset,len(rows),'imágenes; matriz, exactitud y aceptación verificadas')
print('Reserva de 30 por clase y separación de grupos, píxeles y archivos: verificadas')

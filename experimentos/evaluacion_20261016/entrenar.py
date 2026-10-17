"""Nueva transferencia ImageNet sin reutilizar pesos del modelo bovino previo."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import sys,json,time
from pathlib import Path
import numpy as np
import tensorflow as tf
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'05_app')]
from clasificador import preprocesar
from comun.imagenes import abrir_rgb,hash_archivo

def main():
    tf.keras.utils.set_random_seed(42)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    cfg=json.loads((OUT/'config_experimento.json').read_text(encoding='utf-8'))
    manifest=json.loads((OUT/'division/manifiesto.json').read_text(encoding='utf-8'))
    rows=[r for r in manifest['registros'] if r['conjunto']=='entrenamiento']
    labels=[c['etiqueta'] for c in cfg['clases']]
    base=tf.keras.applications.MobileNetV2(input_shape=(224,224,3),alpha=.35,include_top=False,weights='imagenet',pooling='avg')
    base.trainable=False
    xs=[];ys=[];start=time.time()
    for offset in range(0,len(rows),32):
        batch=rows[offset:offset+32]
        images=np.concatenate([preprocesar(abrir_rgb(OUT/'division'/r['destino']),cfg['modelo']['preprocesamiento']) for r in batch])
        xs.append(base(images,training=False).numpy());ys.extend(labels.index(r['clase']) for r in batch)
        if offset%320==0:print(f'Características {min(offset+32,len(rows))}/{len(rows)}',flush=True)
    x=np.concatenate(xs);y=np.array(ys)
    head=tf.keras.Sequential([tf.keras.Input(shape=(x.shape[1],)),tf.keras.layers.Dropout(.2),tf.keras.layers.Dense(5,activation='softmax',kernel_regularizer=tf.keras.regularizers.l2(.0001))])
    head.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    freq=np.bincount(y,minlength=5);weights={i:len(y)/(5*int(freq[i])) for i in range(5)}
    history=[];rng=np.random.default_rng(42)
    # Fixed 40 epochs; test images never used to choose parameters or stopping.
    for epoch in range(40):
        head.reset_metrics();order=rng.permutation(len(y))
        for off in range(0,len(y),32):
            ix=order[off:off+32];metrics=head.train_on_batch(x[ix],y[ix],sample_weight=np.array([weights[int(v)] for v in y[ix]]),return_dict=True)
        history.append({k:float(v) for k,v in metrics.items()})
        print(f'Último lote de época {epoch+1}/40: {history[-1]}',flush=True)
    inputs=tf.keras.Input(shape=(224,224,3),name='imagenes')
    model=tf.keras.Model(inputs,head(base(inputs,training=False),training=False))
    target=OUT/'modelo'
    if target.exists():raise RuntimeError('Modelo ya existente: no sobrescribir.')
    class Export(tf.Module):
        def __init__(self):super().__init__();self.model=model
        @tf.function(input_signature=[tf.TensorSpec([None,224,224,3],tf.float32,name='imagenes')])
        def serving_default(self,imagenes):return {'probabilidades':self.model(imagenes,training=False)}
    obj=Export();tf.saved_model.save(obj,str(target),signatures={'serving_default':obj.serving_default})
    (target/'labels.txt').write_text('\n'.join(f'{i} {l}' for i,l in enumerate(labels))+'\n',encoding='utf-8')
    evidence={'arquitectura':'MobileNetV2','alpha':.35,'pesos_iniciales':'ImageNet de Keras; no se carga el SavedModel bovino anterior','base_congelada':True,'epocas':40,'lote':32,'optimizador':'Adam','tasa':.001,'dropout':.2,'l2':.0001,'semilla':42,'n_entrenamiento':len(y),'pesos_clase':weights,'parametros_totales':model.count_params(),'parametros_entrenables':sum(int(np.prod(v.shape)) for v in head.trainable_weights),'historia':history,'segundos':time.time()-start,'manifiesto_sha256':hash_archivo(OUT/'division/manifiesto.json'),'tensorflow':tf.__version__}
    evidence['historia_ultimo_lote_por_epoca']=evidence.pop('historia')
    (OUT/'entrenamiento.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Modelo exportado',target,flush=True)
if __name__=='__main__':main()

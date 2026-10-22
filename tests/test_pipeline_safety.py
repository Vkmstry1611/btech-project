import importlib.util
from pathlib import Path
import pytest
from catalogbank_ocr.stage2.llm_extractor import ExtractionResult, ExtractedProduct
from catalogbank_ocr.stage2.evidence_validation import validate_evidence


def test_all_explicit_table_skus_required():
    r=ExtractionResult('c','d',[]); r.products=[ExtractedProduct('Imperial',sku='A-1')]
    data={'chunk_id':'c','section_path':[], 'context':{'evidence':[{'page':1,'block_id':'b','row':{'ITEM #':'A-1','METRIC ITEM #':'A-1/M'}}]}}
    assert any(i['code']=='MISSING_TABLE_SKU' and i['value']=='A-1/M' for i in validate_evidence([r],[data]))


def test_failed_run_does_not_leave_stale_final(tmp_path,monkeypatch):
    script=Path(__file__).parents[1]/'scripts/run_final_pipeline.py'
    spec=importlib.util.spec_from_file_location('final_pipeline_test',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    prior=tmp_path/'final'/'test'/'test_final.json';prior.parent.mkdir(parents=True);prior.write_text('{}')
    def fail(*args):raise ValueError('bad source')
    monkeypatch.setattr(module,'_process_pdf',fail)
    with pytest.raises(ValueError): module.process_pdf(Path('test.pdf'),tmp_path,None,None,False)
    assert not prior.exists()
    assert len(list(prior.parent.glob('*previous*')))==1
    assert 'failed' in (prior.parent/'status.json').read_text()


def test_numbered_heading_depth():
    from catalogbank_ocr.semantic.block_types import infer_heading_level
    assert infer_heading_level('paragraph_title','2.3.1 Pumps')==3
    assert infer_heading_level('doc_title','Catalog')==1


def test_spatial_container_never_crosses_page_boundary():
    from catalogbank_ocr.schemas.semantic_block import SemanticDocument,PageSemanticBlocks,SemanticBlock
    from catalogbank_ocr.hierarchy.hierarchy_builder import build_hierarchy_document
    def b(id,page,kind):return SemanticBlock(id=id,page=page,type=kind,source_type=kind,text=id,bbox=[0,0,100,100],confidence=1,order=0,source_ref='test')
    doc=SemanticDocument(pages=[PageSemanticBlocks(page=1,blocks=[b('card',1,'product_card')]),PageSemanticBlocks(page=2,blocks=[b('text',2,'paragraph')])])
    hierarchy=build_hierarchy_document(doc)
    assert not hierarchy.root.children[0].children
    assert hierarchy.root.children[1].page==2


def test_page_batches_preserve_input_and_output_alignment(tmp_path):
    import json
    from catalogbank_ocr.ocr.ppstructure import run_ppstructurev3_batch
    class Prediction:
        def __init__(self,path): self.path=path
        def save_to_json(self,save_path):
            (Path(save_path)/(Path(self.path).stem+'.json')).write_text(json.dumps({'input_path':self.path}))
    class Engine:
        def predict(self,paths):
            assert isinstance(paths,list)
            return [Prediction(path) for path in paths]
    paths=[tmp_path/'p1.png',tmp_path/'p2.png']
    for path in paths: path.write_bytes(b'test')
    outputs=run_ppstructurev3_batch(paths,[tmp_path/'one',tmp_path/'two'],Engine(),False)
    assert [json.loads(r.json_files[0].read_text())['input_path'] for r in outputs]==[str(p) for p in paths]


def test_full_multipage_pipeline_persists_correct_raw_page_provenance(tmp_path):
    import json
    import fitz
    from catalogbank_ocr.research_stage1_pipeline import run_stage1_for_pdf_multipage
    from catalogbank_ocr.preprocessing.image_preprocessing import ImagePreprocessingConfig
    fixture=json.loads((Path(__file__).parent/'fixtures/V21_1_Optomechanics_45_res.json').read_text(encoding='utf-8'))
    source=tmp_path/'two_pages.pdf'
    with fitz.open() as pdf:
        for number in (1,2): pdf.new_page().insert_text((72,72),f'Page {number}')
        pdf.save(source)
    class Prediction:
        def __init__(self,path): self.path=path
        def save_to_json(self,save_path):
            data=dict(fixture,input_path=self.path)
            (Path(save_path)/'result.json').write_text(json.dumps(data),encoding='utf-8')
    class Engine:
        _catalogbank_runtime={'device':'test'}
        def predict(self,path): return [Prediction(path)]
    dest=tmp_path/'out'
    result=run_stage1_for_pdf_multipage(source,dest,ImagePreprocessingConfig(),Engine(),debug_artifacts=False)
    assert result.total_pages==2
    assert [json.loads((dest/'raw'/'two_pages'/f'page_{n:03d}'/'result.json').read_text(encoding='utf-8'))['page_index'] for n in (1,2)]==[0,1]
    metrics=json.loads((dest/'metrics'/'two_pages'/'stage1.json').read_text(encoding='utf-8'))
    assert metrics['state']=='complete' and len(metrics['pages'])==2
    assert metrics['runtime']['device']=='test'

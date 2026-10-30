import json
from pathlib import Path
from catalogbank_ocr.stage2.context_builder import build_context_chunks
from catalogbank_ocr.stage2.llm_input_builder import build_complete_inputs
from catalogbank_ocr.stage2.llm_extractor import ExtractedProduct, ExtractedAttribute, ExtractionResult, _parse_llm_response
from catalogbank_ocr.stage2.normalization import normalize_batch, normalize_sku
from catalogbank_ocr.stage2.catalog_export import build_catalog_json, validate_catalog
from catalogbank_ocr.stage2.validation import validate_extraction
from catalogbank_ocr.stage2.table_parser import _parse_html_table


def block(text, kind='paragraph', page=1, order=0):
    return dict(id=f'p{page}b{order}', text=text, type=kind, page=page, order=order)


def test_page_roles_and_continuation():
    doc = {'pages': [
        {'page':1,'blocks':[block('Acme Catalog 2026','heading')]},
        {'page':2,'blocks':[block('Contents','heading',2),block('Pumps .... 3\nValves .... 9',page=2,order=1)]},
        {'page':3,'blocks':[block('Pumps','heading',3)]},
        {'page':4,'blocks':[block('SKU Ab-01\nFull second line','product_card',4)]},
        {'page':5,'blocks':[block('Continued specifications\nPressure 10 bar',page=5)]}]}
    chunks = build_context_chunks(doc)
    assert [p['classification']['role'] for p in doc['pages']] == ['cover','index','section_divider','product','uncertain']
    assert len(chunks) == 1 and chunks[0].section_path == ['Pumps']
    assert chunks[0].paragraphs[0].page == 5
    assert 'Full second line' in chunks[0].product_cards[0].text


def test_unknown_page_is_not_discarded():
    doc={'pages':[{'page':1,'blocks':[block('An unusual uncategorized paragraph')]}]}
    assert build_context_chunks(doc)[0].paragraphs[0].text.startswith('An unusual')


def test_no_truncation_and_provenance():
    html='<table><tr><th>SKU</th><th>Price</th></tr>'+''.join(f'<tr><td>A{i}</td><td>{i}</td></tr>' for i in range(80))+'</table>'
    doc={'pages':[{'page':1,'blocks':[block('Products','heading'),block(html,'table',order=1),block('Evidence '*1200,order=2)]}]}
    inputs=build_complete_inputs(build_context_chunks(doc),max_chars=4000)
    evidence=[r for x in inputs for r in x['context']['evidence']]
    assert sum('row' in r for r in evidence)==80
    assert any(r.get('row',{}).get('SKU')=='A79' for r in evidence)
    assert all('page' in r and 'block_id' in r for r in evidence)
    assert max(r.get('offset',0) for r in evidence)>600


def test_description_exact_sku_variants_and_stable_export():
    raw=json.dumps({'products':[{'name':'Red pump','model':'P-1','sku':'Ab 01','description':'Red finish','category':'Pumps'}, {'name':'Blue pump','model':'P-1','sku':'Ab/02','category':'Pumps'}]})
    result=_parse_llm_response(raw,'c1','catalog',['Pumps'])
    normalized=normalize_batch([result,result])
    catalog=build_catalog_json({},normalized,pdf_path='catalog.pdf')
    model=catalog['subcategories'][0]['models'][0]
    assert model['description']=='Red finish'
    assert len(model['sellableItems'])==2
    assert model['sellableItems'][0]['articleNumber']=='Ab 01'
    assert normalize_sku(' Ab 01 ')=='Ab 01'
    assert catalog==build_catalog_json({},normalized,pdf_path='catalog.pdf')
    validate_catalog(catalog)


def test_orphan_and_conflicting_attributes():
    r=ExtractionResult('c','doc',[])
    r.attributes=[ExtractedAttribute('missing','Size','10 mm')]
    assert any(i.code=='ORPHANED_ATTRIBUTE' for i in validate_extraction(r).issues)
    r.products=[ExtractedProduct('missing',sku='x')]
    r.attributes.append(ExtractedAttribute('missing','Size','20 mm'))
    normalized=normalize_batch([r])
    assert len(normalized.attributes)==2
    item=build_catalog_json({},normalized)['subcategories'][0]['models'][0]['sellableItems'][0]
    assert item['properties']['Size']==['10.0 mm','20.0 mm']


def test_merged_table_cells_and_duplicate_headers():
    headers,rows=_parse_html_table('<table><tr><th>SKU</th><th>Size</th><th>Size</th></tr><tr><td rowspan="2">A</td><td>1</td><td>2</td></tr><tr><td>3</td><td>4</td></tr></table>')
    assert len(set(headers))==3
    assert rows[1]['SKU']=='A' and rows[1]['Size [3]']=='4'


def test_sample_schema_is_accepted():
    validate_catalog(json.loads((Path(__file__).parents[1]/'sampleoutput.json').read_text(encoding='utf-8')))


def test_codes_and_complex_dimensions_not_parsed_as_units():
    from catalogbank_ocr.stage2.normalization import normalize_attribute_value, parse_value_unit
    for value in ('MS12B', '4-40 (M3 x 0.5)', '12" x 12"', '1/2" (12.5 mm)'):
        assert parse_value_unit(value) is None
        assert normalize_attribute_value(value)[0] == value


def test_grounding_rejects_heading_and_invented_sku():
    from catalogbank_ocr.stage2.evidence_validation import validate_evidence
    r=ExtractionResult('c','doc',['Pumps'])
    r.products=[ExtractedProduct('Pumps'),ExtractedProduct('Valve',sku='invented')]
    request={'chunk_id':'c','section_path':['Pumps'],'context':{'evidence':[{'page':1,'block_id':'b','text':'SKU A-1'}]}}
    assert {v['code'] for v in validate_evidence([r],[request])} == {'HEADING_AS_PRODUCT','UNSUPPORTED_IDENTIFIER'}


def test_same_sku_aliases_and_repeated_prices():
    a=ExtractionResult('a','doc',[]);b=ExtractionResult('b','doc',[])
    a.products=[ExtractedProduct('First name',sku='A 1')]
    b.products=[ExtractedProduct('Other name',sku='A 1')]
    a.attributes=[ExtractedAttribute('First name','Price','10','USD')]
    b.attributes=[ExtractedAttribute('Other name','Price','10','USD')]
    norm=normalize_batch([a,b]);doc=build_catalog_json({},norm)
    assert len(norm.products)==1
    assert len(doc['subcategories'][0]['models'][0]['sellableItems'][0]['listPrices'])==1


def test_page_annotation_is_idempotent():
    from catalogbank_ocr.stage2.page_roles import annotate_pages
    doc={'pages':[{'page':1,'blocks':[block('Contents','heading'),block('Pumps ... 2\nValves ... 3')]}]}
    annotate_pages(doc);first=json.dumps(doc);annotate_pages(doc)
    assert json.dumps(doc)==first


def test_streaming_pixels_match_existing_renderer(tmp_path):
    import fitz
    from PIL import Image, ImageChops
    from catalogbank_ocr.preprocessing.streaming import prepared_pages
    from catalogbank_ocr.preprocessing.image_preprocessing import render_and_preprocess_all_pages, ImagePreprocessingConfig
    pdf=tmp_path/'pages.pdf'
    with fitz.open() as doc:
        for i in range(3):
            page=doc.new_page(width=300,height=400);page.insert_text((20,40),f'SKU A-{i} / 3/8 in')
        doc.save(pdf)
    config=ImagePreprocessingConfig()
    old=render_and_preprocess_all_pages(pdf,tmp_path/'old',tmp_path/'oldprep',config)
    new=list(prepared_pages(pdf,tmp_path/'new',tmp_path/'newprep',config))
    assert len(new)==3
    for a,b in zip(old,new):
        with Image.open(a.processed_image) as x, Image.open(b.processed_image) as y:
            assert ImageChops.difference(x,y).getbbox() is None


def test_variant_attributes_do_not_leak_between_skus():
    r=ExtractionResult('c','d',[])
    r.products=[ExtractedProduct('Pump',model='P',sku='RED'),ExtractedProduct('Pump',model='P',sku='BLUE')]
    r.attributes=[ExtractedAttribute('RED','Color','red'),ExtractedAttribute('BLUE','Color','blue')]
    result=build_catalog_json({},normalize_batch([r]))
    items=result['subcategories'][0]['models'][0]['sellableItems']
    assert {i['articleNumber']:i['properties']['Color'] for i in items} == {'RED':'red','BLUE':'blue'}


def test_prices_preserve_locations_and_decimal_comma():
    from catalogbank_ocr.stage2.catalog_export import parse_price
    assert parse_price('223,59') == 223.59
    assert parse_price('$ 257.00') == 257
    assert parse_price('1,234.56') == parse_price('1.234,56') == 1234.56
    import pytest
    for malformed in ('1,23.45','12.34,56','1.234,5,6'):
        with pytest.raises(ValueError): parse_price(malformed)
    r=ExtractionResult('c','d',[]);r.products=[ExtractedProduct('Chair',sku='A')]
    r.attributes=[ExtractedAttribute('A','Price','79','EUR','Germany'),ExtractedAttribute('A','Price','79','EUR','Sweden')]
    prices=build_catalog_json({},normalize_batch([r]))['subcategories'][0]['models'][0]['sellableItems'][0]['listPrices']
    assert {p['location'] for p in prices} == {'Germany','Sweden'}


def test_large_document_keeps_all_pages_and_shared_context():
    doc={'pages':[{'page':i,'blocks':[block('Pumps','heading',i),block(f'Part number A-{i}','product_card',i,1)]} for i in range(1,61)]}
    chunks=build_context_chunks(doc)
    assert len(doc['pages'])==60
    assert {b.page for c in chunks for b in c.product_cards}==set(range(1,61))
    assert sum(len(c.product_cards) for c in chunks)==60


def test_table_reconciliation_preserves_metric_variant_without_invented_model():
    from catalogbank_ocr.stage2.table_products import reconcile_table_products
    r=ExtractionResult('c','d',['Breadboards'])
    r.products=[ExtractedProduct('Breadboard',sku='MS12B')]
    evidence=[{'page':1,'block_id':'table','row':{'ITEM #':'MS12B','METRIC ITEM #':'MS12B/M','DESCRIPTION':'Breadboard'}}]
    request={'chunk_id':'c','section_path':['Breadboards'],'context':{'evidence':evidence}}
    reconcile_table_products([r],[request])
    assert {p.sku for p in r.products} == {'MS12B','MS12B/M'}
    assert all(p.model is None for p in r.products)
    catalog=build_catalog_json({},normalize_batch([r]))
    models=catalog['subcategories'][0]['models']
    assert len(models)==1 and len(models[0]['sellableItems'])==2


def test_explicit_tables_use_literal_values_and_unsafe_prices_are_reviewed():
    from catalogbank_ocr.stage2.table_products import extract_explicit_table
    request={'chunk_id':'c','doc_stem':'d','section_path':['Pumps'],'context':{'evidence':[
        {'page':1,'block_id':'b','row':{'ITEM #':'A-1','METRIC ITEM #':'A-1/M','DESCRIPTION':'Pump','EUR':'223,59','Material':'Steel'}}]}}
    result=extract_explicit_table(request)
    assert {p.sku for p in result.products} == {'A-1','A-1/M'}
    assert {(a.entity,a.value,a.unit) for a in result.attributes if a.key=='Price'} == {('A-1','223.59','EUR'),('A-1/M','223.59','EUR')}
    request['context']['evidence'][0]['row']['EUR']='223,59 80,21'
    result=extract_explicit_table(request)
    assert result.attributes[1].value=='223,59 80,21'
    issues=[]
    build_catalog_json({},normalize_batch([result]),review_issues=issues)
    assert issues and all(i['code']=='AMBIGUOUS_PRICE' for i in issues)


def test_hybrid_extraction_keeps_every_record_without_resending_clear_rows():
    from catalogbank_ocr.stage2.table_products import extract_with_tables
    class Extractor:
        def extract(self, request):
            assert len(request['context']['evidence'])==1
            assert request['context']['evidence'][0]['text']=='A supporting sentence'
            return ExtractionResult(request['chunk_id'],request['doc_stem'],[])
    request={'chunk_id':'c','doc_stem':'d','section_path':[], 'prompt':'RULES\nEVIDENCE:\n{}',
        'context':{'evidence':[{'page':1,'block_id':'b','row':{'SKU':'A-1','DESCRIPTION':'Pump','USD':'10'}},
                               {'page':1,'block_id':'text','text':'A supporting sentence'}]}}
    result=extract_with_tables(Extractor(),[request])[0]
    assert result.products[0].sku=='A-1'
    assert result.attributes[0].value=='10.0'


def test_graph_attributes_keep_variant_identity_and_conflicting_values():
    from catalogbank_ocr.stage2.llm_extractor import ExtractedRelation
    from catalogbank_ocr.stage2.graph_ingest import _build_statements, _MERGE_PRODUCT, _MERGE_ATTRIBUTE, _MERGE_PRODUCT_ATTR
    r=ExtractionResult('c','d',[])
    r.products=[ExtractedProduct('Pump',sku='a'),ExtractedProduct('Pump',sku='A')]
    r.attributes=[ExtractedAttribute('a','Color','red'),ExtractedAttribute('A','Color','blue'),ExtractedAttribute('a','Color','green')]
    r.relations=[ExtractedRelation('a','COMPATIBLE_WITH','Target'),ExtractedRelation('A','COMPATIBLE_WITH','Target')]
    assert len(normalize_batch([r]).relations)==2
    statements=_build_statements(normalize_batch([r]))
    products=[s.params['id'] for s in statements if s.cypher==_MERGE_PRODUCT]
    attributes=[s.params['id'] for s in statements if s.cypher==_MERGE_ATTRIBUTE]
    links=[s.params['prod_id'] for s in statements if s.cypher==_MERGE_PRODUCT_ATTR]
    assert len(set(products))==2 and len(set(attributes))==3
    assert links[0]==links[2] and links[0]!=links[1]


def test_zero_price_is_preserved_and_malformed_arrays_rejected():
    result=_parse_llm_response(json.dumps({'products':[{'name':'Free item','sku':'A'}], 'attributes':[{'entity':'A','key':'Price','value':0,'unit':'USD'}]}),'c','d',[])
    assert result.attributes[0].value=='0'
    assert build_catalog_json({},normalize_batch([result]))['subcategories'][0]['models'][0]['sellableItems'][0]['listPrices'][0]['amount']==0
    assert _parse_llm_response('{"products":"bad"}','c','d',[]).parse_error


def test_identical_model_and_unrelated_sku_codes_do_not_merge():
    r=ExtractionResult('c','d',[])
    r.products=[ExtractedProduct('Design A',model='X',sku='A'),ExtractedProduct('Unrelated item',sku='X')]
    models=build_catalog_json({},normalize_batch([r]))['subcategories'][0]['models']
    assert len(models)==2


def test_distributor_and_manufacturer_codes_are_one_item_not_two_variants():
    from catalogbank_ocr.stage2.evidence_validation import validate_evidence
    from catalogbank_ocr.stage2.table_products import extract_explicit_table, row_codes
    row={'Mfg. Part No.':'DV164037', 'Description':'Starter kit', 'Stock No.':'25R7377', 'Price Each':'1+'}
    assert row_codes(row)==['25R7377']
    # Also prefer stock number if the manufacturer column uses the generic label.
    assert row_codes({'Part No.':'DV164037', 'Stock No.':'25R7377'})==['25R7377']
    request={'chunk_id':'c','doc_stem':'d','section_path':[], 'context':{'evidence':[{'page':1,'block_id':'b','row':row}]}}
    result=extract_explicit_table(request)
    assert len(result.products)==1 and result.products[0].sku=='25R7377'
    assert result.products[0].model is None
    assert any(a.key=='Mfg. Part No.' and a.value=='DV164037' for a in result.attributes)
    assert any(a.key=='Price Each' and a.value=='1+' for a in result.attributes)
    assert not validate_evidence([result],[request])


def test_repeated_sku_reconciles_variant_rows_across_pages():
    a=ExtractionResult('page1','d',[])
    b=ExtractionResult('page2','d',[])
    a.products=[ExtractedProduct('One',sku='A',variant_group='row-a'),ExtractedProduct('Two',sku='B',variant_group='row-a',description='Basic pump')]
    b.products=[ExtractedProduct('Two',sku='B',variant_group='row-b',description='With pressure sensor'),ExtractedProduct('Three',sku='C',variant_group='row-b')]
    normalized=normalize_batch([a,b])
    assert len(normalized.products)==3
    assert len({p.variant_group for p in normalized.products})==1
    description=next(p.description for p in normalized.products if p.sku=='B')
    assert 'Basic pump' in description and 'With pressure sensor' in description
    models=build_catalog_json({},normalized)['subcategories'][0]['models']
    assert len(models)==1 and len(models[0]['sellableItems'])==3


def test_explicit_model_heading_is_valid_source_context_for_continuation():
    from catalogbank_ocr.stage2.evidence_validation import validate_evidence
    from catalogbank_ocr.stage2.table_products import extract_explicit_table
    result=ExtractionResult('c','d',['Model PX-1'])
    result.products=[ExtractedProduct('Pump',model='PX-1',sku='A-1')]
    request={'chunk_id':'c','doc_stem':'d','section_path':['Model PX-1'],'context':{'evidence':[{'page':2,'block_id':'b','row':{'SKU':'A-1','Description':'Pump'}}]}}
    assert not validate_evidence([result],[request])
    assert extract_explicit_table(request).products[0].model=='PX-1'
    request['section_path']=['PX-1']
    assert extract_explicit_table(request).products[0].model is None


def test_bottom_ocr_headers_preserve_preceding_rows_and_price_labels():
    table='<table><tr><td>15.22</td><td>85W3986</td><td>AM3359</td></tr><tr><td>1.93</td><td>73W9706</td><td>AM3357</td></tr><tr><td>1-4+</td><td>Stock No.</td><td>Mfg. Part No.</td></tr><tr><td>Price Each</td><td></td><td></td></tr></table>'
    headers,rows=_parse_html_table(table)
    assert len(rows)==2
    assert [row['Stock No.'] for row in rows]==['85W3986','73W9706']
    assert rows[0]['Mfg. Part No.']=='AM3359'
    assert 'Price Each' in headers[0] and '1-4+' in headers[0]


def test_mcmaster_fixture_extracts_literal_codes_and_adjacent_prices():
    from catalogbank_ocr.stage2.table_products import extract_inline_table_row, inline_part_prices
    raw=json.loads((Path(__file__).parent/'fixtures/mcmaster-125_3378_15_res.json').read_text(encoding='utf-8'))
    table=next(b['block_content'] for b in raw['parsing_res_list'] if b['block_label']=='table')
    _,rows=_parse_html_table(table)
    request={'chunk_id':'c','doc_stem':'d','section_path':['Thread Adapters']}
    record={'page':15,'block_id':'table','row':rows[1]}
    result=extract_inline_table_row(request,record)
    assert {p.sku for p in result.products}>={'1456N251','1456N297','1456N157','1456N204'}
    assert {a.entity:a.value for a in result.attributes}['1456N204']=='18.38'
    assert all(a.unit is None for a in result.attributes)
    assert not inline_part_prices({'a':'1456N251 $18.38 $16.22'})
    assert not inline_part_prices({'a':'M10×1.252684K073 15.77'})
    assert extract_inline_table_row(request,{'page':1,'block_id':'b','row':{'Mfg. Part No.':'1456N251 $18.38','Stock No.':''}}) is None


def test_product_family_stays_above_specifications_and_smaller_subsection():
    def entry(index,text,kind,width,top):
        return {'id':f'b{index}','page':1,'type':kind,'source_type':'paragraph_title' if kind=='heading' else 'table',
                'text':text,'heading_level':2 if kind=='heading' else None,'bbox':[0,top,width,top+30],
                'order':index,'raw':{'block_content':text}}
    document={'source':{'pdf_path':'test.pdf'},'pages':[{'page':1,'blocks':[
        entry(1,'ScienceDesk Accessories','heading',550,100),
        entry(2,'Under Shelves','heading',220,500),
        entry(3,'<table><tr><td>SKU</td><td>Price</td></tr><tr><td>PSY300</td><td>$80</td></tr></table>','table',600,700),
        entry(4,'ScienceDesk Casters','heading',410,1100),
        entry(5,'<table><tr><td>SKU</td><td>Price</td></tr><tr><td>PSY100</td><td>$150</td></tr></table>','table',600,1300),
    ]}]}
    paths=[c.section_path for c in build_context_chunks(document) if c.tables]
    assert ['ScienceDesk Accessories','Under Shelves'] in paths
    assert ['ScienceDesk Casters'] in paths


def test_oversized_table_row_preserves_all_cells_in_bounded_requests():
    from catalogbank_ocr.stage2.context_builder import ContextChunk,RawBlock
    html='<table><tr><td>SKU</td><td>Details</td></tr><tr><td>A-1</td><td>'+('A'*5000+'B'*5000)+'</td></tr></table>'
    chunk=ContextChunk('c','d',1,['Pumps'],1)
    chunk.tables=[RawBlock('t',1,'table','table',html,None,None,0,None,{'block_content':html})]
    requests=build_complete_inputs([chunk],max_chars=6000)
    fragments=[v for r in requests for v in r['context']['evidence'] if 'row' in v]
    detail=[v['row']['Details'] for v in fragments if 'Details' in v['row']]
    assert len(requests)>1 and all(len(json.dumps(v))<=6000 for v in fragments)
    assert 'A'*3000 in ''.join(detail) and 'B'*3000 in ''.join(detail)


def test_dense_ocr_paragraph_splits_by_complete_lines():
    from catalogbank_ocr.stage2.context_builder import ContextChunk,RawBlock
    lines=[f'1/8.4464K{i:03d}$3.85 4464K{i+100:03d}$2.26\n' for i in range(30)]
    text=''.join(lines)
    chunk=ContextChunk('c','d',1,['Fittings'],1)
    chunk.paragraphs=[RawBlock('b',1,'text','text',text,None,None,0,None,{'block_content':text})]
    requests=build_complete_inputs([chunk])
    parts=[e['text'] for r in requests for e in r['context']['evidence'] if e['block_id']=='b']
    assert len(parts)>1 and all(len(p)<=700 for p in parts)
    assert ''.join(parts)==text
    assert all(sum(e.get('block_id')=='b' for e in r['context']['evidence'])==1 for r in requests)


def test_unsupported_family_entries_are_removed_without_losing_real_skus():
    from catalogbank_ocr.stage2.evidence_validation import remove_heading_only_products, validate_evidence
    result=ExtractionResult('c','d',['PerformancePlus Series'])
    result.products=[ExtractedProduct('PerformancePlus Series',description='General feature'),
                     ExtractedProduct('PBI11106',sku='PBI11106')]
    result.attributes=[ExtractedAttribute('PerformancePlus Series','Feature','Stable'),
                       ExtractedAttribute('PBI11106','Size','Small')]
    request={'chunk_id':'c','section_path':['PerformancePlus Series'],
             'context':{'evidence':[{'page':1,'block_id':'b','text':'PBI11106'}]}}
    notes=remove_heading_only_products([result],[request])
    assert [p.sku for p in result.products]==['PBI11106']
    assert [a.entity for a in result.attributes]==['PBI11106']
    assert notes[0]['code']=='HEADING_FAMILY_IGNORED'
    assert not [v for v in validate_evidence([result],[request]) if v['severity']=='error']


def test_displaced_skus_are_kept_without_inheriting_previous_row_price():
    from catalogbank_ocr.stage2.table_products import extract_explicit_table, reconcile_table_products, spillover_row_codes
    from catalogbank_ocr.stage2.evidence_validation import validate_evidence
    row={'ITEM #*':'PBH11112/NH','METRIC ITEM #':'PBH51512/NH','$':'$985.00',
         'col_7':'PBH11115/NH','col_8':'PBH51515/NH'}
    request={'chunk_id':'c','doc_stem':'d','section_path':['Breadboards'],
             'context':{'evidence':[{'page':1,'block_id':'t','row':row}]}}
    assert spillover_row_codes(row)==['PBH11115/NH','PBH51515/NH']
    result=extract_explicit_table(request)
    reconcile_table_products([result],[request])
    assert {p.sku for p in result.products}==set(row.values())-{'$985.00'}
    assert not any(a.entity in {'PBH11115/NH','PBH51515/NH'} and a.key=='Price' for a in result.attributes)
    assert [i['code'] for i in validate_evidence([result],[request])].count('TABLE_SKU_PRICE_NOT_RECOVERED')==2


def test_generic_unpriced_item_is_not_a_sellable_product():
    from catalogbank_ocr.stage2.evidence_validation import remove_heading_only_products
    result=ExtractionResult('c','d',['Under Shelves'])
    result.products=[ExtractedProduct('Under Shelf',description='Shelf feature'),ExtractedProduct('PSY300',sku='PSY300')]
    request={'chunk_id':'c','section_path':['Under Shelves'],
             'context':{'evidence':[{'page':1,'block_id':'b','text':'Under Shelf holds PSY300'}]}}
    notes=remove_heading_only_products([result],[request])
    assert [p.sku for p in result.products]==['PSY300']
    assert any(n['code']=='UNSUPPORTED_ITEM_IGNORED' for n in notes)


def test_native_text_pairs_only_exact_dotted_sku_prices():
    from catalogbank_ocr.stage2.native_text import paired_codes_in_text
    source='Size 1/8.. . 4464K211.. . . $3.85\n4464K221.. . . 2.26\nSee 4464K222 for details\n'
    assert paired_codes_in_text(source, 2)==[
        {'page':2,'sku':'4464K211','price':'3.85','currency':'USD'},
        {'page':2,'sku':'4464K221','price':'2.26','currency':None},
    ]
    assert paired_codes_in_text('M10×1.252684K073 15.77', 1)==[]


def test_material_grade_is_not_a_model_or_unsold_product():
    from catalogbank_ocr.stage2.evidence_validation import remove_heading_only_products, validate_evidence
    result=ExtractionResult('c','d',['Thread Adapters'])
    result.products=[ExtractedProduct('Reducing Thread Adapters',model='316 Stainless Steel'),
                     ExtractedProduct('1456N251',model='316 Stainless Steel',sku='1456N251')]
    request={'chunk_id':'c','section_path':['Thread Adapters'],
             'context':{'evidence':[{'page':1,'block_id':'b','text':'316 Stainless Steel 1456N251'}]}}
    notes=remove_heading_only_products([result],[request])
    assert [p.sku for p in result.products]==['1456N251']
    assert result.products[0].model is None
    assert any(n['code']=='UNSUPPORTED_ITEM_IGNORED' for n in notes)
    assert not [v for v in validate_evidence([result],[request]) if v['severity']=='error']


def test_native_sku_supplement_keeps_exact_identity_and_family_group(monkeypatch):
    from catalogbank_ocr.stage2 import native_text
    monkeypatch.setattr(native_text, 'paired_native_codes', lambda _: [
        {'page':1,'sku':'4464K211','price':'3.85','currency':'USD'},
        {'page':1,'sku':'4464K221','price':'2.26','currency':None},
    ])
    result=ExtractionResult('c','catalog',['Pipe Fittings'])
    result.products=[ExtractedProduct('Existing',sku='4464K211')]
    request={'chunk_id':'c','section_path':['Pipe Fittings'],
             'context':{'evidence':[{'page':1,'block_id':'b','text':'4464K221'}]}}
    results,inputs=[result],[request]
    added,source=native_text.supplement_missing_skus(Path('catalog.pdf'),results,inputs)
    assert len(added)==1 and source=={'4464K211','4464K221'}
    assert len(results)==2 and results[-1].products[0].sku=='4464K221'
    assert results[-1].products[0].name=='Pipe Fittings'
    assert results[-1].products[0].model is None
    assert results[-1].products[0].variant_group
    assert inputs[-1]['context']['evidence'][0]['row']=={'SKU':'4464K221','Price':'2.26'}


def test_native_optical_variant_row_requires_printed_currency_and_price():
    from catalogbank_ocr.stage2.native_text import ALPHA_SKU_PRICE_ROW
    text='PBH11115/NH\nPBH51515/NH\n$\n1,054.00\n'
    match=ALPHA_SKU_PRICE_ROW.search(text)
    assert match and match.group('codes').splitlines()==['PBH11115/NH','PBH51515/NH']
    assert match.group('amount')=='1,054.00'
    assert ALPHA_SKU_PRICE_ROW.search('PBH11115/NH\nPBH51515/NH\n1,054.00\n') is None

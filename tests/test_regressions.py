import json
from pathlib import Path
from datetime import date
from dataclasses import replace
from decimal import Decimal
import pytest
from pydantic import ValidationError
import actions, policy, store
from actions import ActionContext, Status
from models import ReturnReason as RR, ActionType, OrderItem
from evals import run_evals as ev


def ctx(path, who='C003'):
    return ActionContext(store.find_customer(who), store.load_catalog(), store.load_state(path), path, date(2026,11,9))


def test_two_order_lines_are_independent_and_repeats_are_exact(tmp_path):
    path=tmp_path/'state.json'; order=store.load_orders()['ORD-1006']
    a=actions.refund(ctx(path),order,'HOOD-GRY-L',RR.CHANGED_MIND)
    b=actions.refund(ctx(path),order,'CAP-NVY-OS',RR.CHANGED_MIND)
    c=actions.refund(ctx(path),order,'CAP-NVY-OS',RR.CHANGED_MIND)
    assert [a.status,b.status,c.status] == [Status.EXECUTED,Status.EXECUTED,Status.ALREADY_DONE]
    assert b.record == c.record and b.record.sku == 'CAP-NVY-OS'
    assert sorted(r.detail['amount'] for r in store.load_state(path).executed_actions.values()) == ['20.00','60.00']


def test_refund_then_exchange_other_line_succeeds_same_line_denied(tmp_path):
    path=tmp_path/'state.json'; order=store.load_orders()['ORD-1006']
    actions.refund(ctx(path),order,'CAP-NVY-OS',RR.CHANGED_MIND)
    out=actions.exchange(ctx(path),order,'HOOD-GRY-L','HOOD-GRY-M',RR.WRONG_SIZE)
    assert out.status is Status.EXECUTED
    denied=actions.refund(ctx(path),order,'HOOD-GRY-L',RR.CHANGED_MIND)
    assert denied.status is Status.DENIED
    assert len(store.load_state(path).executed_actions)==2


def test_old_key_record_still_prevents_duplicate(tmp_path):
    path=tmp_path/'state.json'; order=store.load_orders()['ORD-1001']
    first=actions.refund(ctx(path,'C001'),order,'TEE-BLK-M',RR.CHANGED_MIND)
    state=store.load_state(path); rec=state.executed_actions.pop(first.record.idempotency_key)
    old='C001|ORD-1001|refund'; state.executed_actions[old]=rec.model_copy(update={'idempotency_key':old})
    store.save_state(path,state)
    out=actions.refund(ctx(path,'alice.tan@example.com'),order,'TEE-BLK-M',RR.CHANGED_MIND)
    assert out.status is Status.ALREADY_DONE and len(store.load_state(path).executed_actions)==1


@pytest.mark.parametrize('order_id,sku,who,new_flag,allowed',[
    ('ORD-1003','JKT-OLV-M','C002',False,False),
    ('ORD-1001','TEE-BLK-M','C001',True,True),
])
def test_catalog_sale_changes_do_not_rewrite_purchase_terms(order_id,sku,who,new_flag,allowed):
    catalog=store.load_catalog(); catalog[sku]=catalog[sku].model_copy(update={'on_sale':new_flag,'sale_name':'New sale' if new_flag else None})
    decision=policy.check_refund(store.find_customer(who),store.load_orders()[order_id],sku,reason=RR.CHANGED_MIND,today=date(2026,11,9),catalog=catalog)
    assert decision.allowed is allowed


def test_verified_defective_sale_refunds_once(tmp_path):
    path=tmp_path/'state.json'; order=store.load_orders()['ORD-1010']
    a=actions.refund(ctx(path,'C002'),order,'JKT-OLV-M',RR.DEFECTIVE,'zip broken')
    b=actions.refund(ctx(path,'C002'),order,'JKT-OLV-M',RR.DEFECTIVE,'zip broken')
    assert a.status is Status.EXECUTED and a.record.detail['amount']=='84.00'
    assert b.status is Status.ALREADY_DONE


def test_model_cannot_supply_verified_defect():
    from agent import RefundIn
    with pytest.raises(ValidationError):
        RefundIn.model_validate({'order_id':'ORD-1003','sku':'JKT-OLV-M','reason':'defective','defect_verified':True})


def test_purchase_sale_status_is_required():
    with pytest.raises(ValidationError):
        OrderItem(sku='X',quantity=1,unit_price=Decimal('5.00'))


def test_eval_persists_completed_cases_on_interrupt(tmp_path,monkeypatch):
    monkeypatch.setattr(ev,'RUNS_DIR',tmp_path)
    monkeypatch.setenv('ANTHROPIC_API_KEY','fake-not-used')
    calls=[]
    def fake(case):
        calls.append(case.id)
        if len(calls)==2: raise KeyboardInterrupt
        return {'id':case.id,'name':case.name,'passed':True,'failures':[]}
    monkeypatch.setattr(ev,'run_case',fake)
    assert ev.main_cli(['01','03'])==130
    report=json.loads(next(tmp_path.glob('*.json')).read_text())
    assert report['status']=='interrupted' and report['summary']['passed']==1
    assert report['active_case']=='03' and report['finished_at']


def test_eval_rejects_unknown_case_and_saves_setup_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(ev,'RUNS_DIR',tmp_path)
    monkeypatch.setenv('ANTHROPIC_API_KEY','fake-not-used')
    assert ev.main_cli(['typo'])==2
    report=json.loads(next(tmp_path.glob('*.json')).read_text())
    assert report['status']=='setup_error'


def test_eval_no_turns_cannot_pass():
    run=ev.CaseRun([],set(),[])
    assert any(not ok for _,ok,_ in ev.evaluate(ev.CASES[-1],run))


def test_refund_eval_rejects_wrong_amount(tmp_path):
    path=tmp_path/'state.json'
    record=actions.refund(ctx(path,'C001'),store.load_orders()['ORD-1001'],'TEE-BLK-M',RR.CHANGED_MIND).record
    bad=record.model_copy(update={'detail':{'amount':'999.00'}})
    assert not ev.exact_refunds([('TEE-BLK-M','25.00')])(ev.CaseRun([],set(),[bad]))[0]

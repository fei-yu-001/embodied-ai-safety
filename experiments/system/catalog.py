"""Read the installed task and scene catalogue without importing a simulator."""
import ast
from functools import lru_cache
from pathlib import Path
import re
import warnings

ROBOCASA = Path(__file__).resolve().parents[1] / 'envs/robocasa/robocasa'
LABELS = {
    'PickPlaceCounterToCabinet': '台面物体放入橱柜', 'PickPlaceCabinetToCounter': '橱柜物体取到台面',
    'PickPlaceCounterToSink': '台面物体放入水槽', 'PickPlaceSinkToCounter': '水槽物体取到台面',
    'PickPlaceCounterToMicrowave': '台面物体放入微波炉', 'PickPlaceMicrowaveToCounter': '微波炉物体取到台面',
    'OpenCabinet': '打开橱柜', 'CloseCabinet': '关闭橱柜', 'OpenDrawer': '打开抽屉', 'CloseDrawer': '关闭抽屉',
    'OpenMicrowave': '打开微波炉', 'CloseMicrowave': '关闭微波炉', 'OpenFridge': '打开冰箱',
    'CloseFridge': '关闭冰箱', 'TurnOnStove': '打开炉灶', 'TurnOffStove': '关闭炉灶',
    'TurnOnSinkFaucet': '打开水龙头', 'TurnOffSinkFaucet': '关闭水龙头', 'PrepareCoffee': '准备咖啡',
    'WashFruit': '清洗水果', 'OpenDishwasher': '打开洗碗机', 'CloseDishwasher': '关闭洗碗机',
}


def _constants(node, resolve=None, owner=None):
    try:
        return list(ast.literal_eval(node))
    except (ValueError, TypeError):
        if isinstance(node, ast.Name) and resolve and owner:
            return list(resolve(owner, node.id))
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and resolve:
            return list(resolve(node.value.id, node.attr))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'list', 'tuple'}:
            return _constants(node.args[0], resolve, owner)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'range':
            return list(range(*(ast.literal_eval(arg) for arg in node.args)))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return _constants(node.left, resolve, owner) + _constants(node.right, resolve, owner)
        return []


@lru_cache(maxsize=1)
def get_catalog():
    registry = ast.parse((ROBOCASA / '__init__.py').read_text(encoding='utf-8'))
    classes = {}
    for path in (ROBOCASA / 'environments/kitchen').rglob('*.py'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', (DeprecationWarning, SyntaxWarning))
            source = ast.parse(path.read_text(encoding='utf-8'))
        for node in source.body:
            if isinstance(node, ast.ClassDef):
                classes[node.name] = node

    @lru_cache(maxsize=None)
    def restrictions(name, field):
        node = classes.get(name)
        if node is None:
            return ()
        for statement in node.body:
            if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == field for t in statement.targets):
                return tuple(_constants(statement.value, restrictions, name))
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in classes:
                return restrictions(base.id, field)
        return ()

    tasks = []
    for statement in registry.body:
        if not isinstance(statement, ast.ImportFrom):
            continue
        module = statement.module or ''
        group = 'atomic' if '.kitchen.atomic.' in module else 'composite' if '.kitchen.composite.' in module else None
        if group is None:
            continue
        for entry in statement.names:
            name = entry.name
            if name not in classes or name in {'OpenDoor', 'CloseDoor'}:
                continue
            doc = ast.get_docstring(classes[name]) or ''
            tasks.append({'id': f'robocasa/{name}', 'name': name,
                          'label': LABELS.get(name, re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name)),
                          'group': group, 'description': doc.split('\n\n')[0].strip()[:400],
                          'excluded_layouts': list(restrictions(name, 'EXCLUDE_LAYOUTS')),
                          'excluded_styles': list(restrictions(name, 'EXCLUDE_STYLES'))})
    tasks.sort(key=lambda item: (item['group'], item['name']))
    def scenes(kind, prefix, label):
        items = []
        for path in (ROBOCASA / f'models/assets/scenes/kitchen_{kind}').rglob(f'{prefix}[0-9][0-9][0-9].yaml'):
            index = int(path.stem[len(prefix):])
            items.append({'id': index, 'label': f'{label} {index:03d}', 'group': path.parent.name})
        return sorted(items, key=lambda x: x['id'])
    layouts = scenes('layouts', 'layout', '布局')
    styles = scenes('styles', 'style', '风格')
    return {'tasks': tasks, 'layouts': layouts, 'styles': styles,
            'counts': {'tasks': len(tasks), 'layouts': len(layouts), 'styles': len(styles)},
            'note': '目录来自本机安装资产。任务会限制可用布局/风格；组合数量不等于已验证场景数量。'}


def task_ids():
    return tuple(item['id'] for item in get_catalog()['tasks'])


def layout_ids():
    return tuple(item['id'] for item in get_catalog()['layouts'])


def style_ids():
    return tuple(item['id'] for item in get_catalog()['styles'])


def task_info(task_id):
    return next((item for item in get_catalog()['tasks'] if item['id'] == task_id), None)

"""Adapt the asset exporter's full manifest to the small scene IPC descriptor."""
from __future__ import annotations

from server.runtime.adapters.visual_assets import export_visual_assets


def attach_visual_assets(geometry, model, directory):
    raw = getattr(model, '_model', model)
    source = export_visual_assets(raw, directory)
    manifest = {'version': 'prism_visual_assets/1', 'asset_key': source['asset_key'],
                'meshes': {}, 'textures': {}, 'materials': {},
                'unsupported': source.get('unsupported', [])[:24],
                'unsupported_count': len(source.get('unsupported', []))}
    for key, item in source['meshes'].items():
        manifest['meshes'][key] = {'file': item['file'], 'triangles': item['triangles'], 'has_uv': item.get('uv_complete', False)}
    for key, item in source['textures'].items():
        manifest['textures'][key] = {name: item[name] for name in ('file', 'width', 'height', 'type', 'cube') if name in item}
    for key, item in source['materials'].items():
        roles = item.get('textures', {})
        texture = roles.get('rgba', roles.get('rgb'))
        manifest['materials'][key] = {'rgba': item.get('rgba') or [1, 1, 1, 1],
            'repeat': item.get('texrepeat') or [1, 1], 'texuniform': item.get('texuniform', False),
            'shininess': item.get('shininess') or 0, 'specular': item.get('specular') or 0,
            'texture': str(texture) if texture is not None else None}
    by_id = {row['geom']: row for row in source['geoms']}
    entries = list(geometry.get('robot', {}).get('geoms', [])) + list(geometry.get('world', {}).get('geoms', []))
    for item in geometry.get('objects', []):
        entries.extend(item.get('geoms', []))
    actual_meshes = approximate = 0
    for entry in entries:
        row = by_id.get(int(entry['geom']))
        if not row:
            approximate += int(entry.get('kind') == 'mesh')
            continue
        entry['texture_support'] = row.get('texture_support')
        if row.get('material') is not None:
            entry['material_id'] = str(row['material'])
        if row.get('mesh_exported'):
            entry['mesh_id'] = str(row['mesh'])
            entry['representation'] = 'visual_mesh'
            actual_meshes += 1
        elif entry.get('kind') == 'mesh':
            approximate += 1
        material = manifest['materials'].get(str(row.get('material')))
        if material:
            texture = manifest['textures'].get(material.get('texture'))
            support = row.get('texture_support')
            supported = bool(texture and (support in ('mesh_uv', 'planar_projection', 'box_cube_face_lookup') or texture.get('cube', {}).get('uniform')))
            if supported or entry.get('colour_source') != 'material_texture':
                entry['colour'] = material['rgba'][:3]
                entry['alpha'] = material['rgba'][3]
    geometry['assets'] = manifest
    geometry.setdefault('provenance', {}).update({
        'representation': 'visual_mesh + primitives' if actual_meshes else 'primitives + mesh_bbox',
        'mesh_assets': len(manifest['meshes']), 'actual_mesh_geoms': actual_meshes,
        'approximate_mesh_geoms': approximate, 'textures_exported': len(manifest['textures']),
        'asset_unsupported_count': manifest['unsupported_count'],
        'note': '真实视觉网格取自编译 MuJoCo 模型；原始纹理独立导出，场景载荷仅含引用；浏览器载入失败或不支持映射时明确降级。',
        'texture_colour': 'authored_texture_when_supported',
    })
    return geometry

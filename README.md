# comfyui-llm-nodes

Кастомные ноды ComfyUI для корпоративного учёта AI-генерации: все вызовы идут через LiteLLM-шлюз компании под персональным виртуальным ключом. ТЗ: `context/comfy-server-platform/nodes-spec.md`.

## Установка (сотрудник)

1. ComfyUI Manager → Install via Git URL → `https://github.com/nakarandashe27/comfyui-llm-nodes` (или `git clone` в `custom_nodes/`).
2. Рядом с пакетом скопировать `config.ini.example` → `config.ini`, вписать адрес шлюза и свой ключ (выдаёт админ).
3. Перезапустить ComfyUI — в категории **LLM** появятся ноды `LLM Text`, `LLM Image`, `LLM Video`.

Ключ живёт только в `config.ini` (он в .gitignore) — в сохранённые воркфлоу и PNG он не попадает, файлами можно делиться свободно.

## Поле project

Необязательное поле в каждой ноде: если заполнено — расход в дашборде виден в разрезе этого проекта (тег `project:<имя>`).

## Проверка

`python test_api.py` — офлайн-самопроверка HTTP-слоя. Приёмка с живым шлюзом — nodes-spec §5.

# AI-Pretest MesSage · AIPM 3.0

Отдельное Streamlit-приложение для оценки рекламного ролика по трём компонентам:

- **AIPM 1.0** — заметность;
- **AIPM 2.0** — запоминаемость;
- **Message Delivery** — считываемость главной идеи.

Итоговый AIPM 3.0 — нормированное произведение трёх индексов. Вместо генеративных
рекомендаций приложение показывает, что в ролике работает и что ограничивает
результат, с детализацией по бизнес-группам признаков.

## Что находится в репозитории

```text
streamlit_app.py                # вход и навигация
app_pages/video_pretest.py      # интерфейс видео-претеста
aipm3/                          # inference и бизнес-интерпретация
artifacts.py                    # загрузка замороженных моделей из Secrets
scripts/build_local_artifacts.py
requirements.txt
packages.txt                    # системный ffmpeg
```

Модели, обучающие таблицы и API-ключи в GitHub не хранятся. Три замороженных
артефакта передаются через Streamlit Secrets в base64.

## Локальный запуск

1. Создать окружение и установить `requirements-dev.txt`.
2. Собрать локальные артефакты командой `scripts/build_local_artifacts.py`, передав
   пути к обучающему parquet AIPM 1.0, модели AIPM 2.0, bundle Message Delivery и
   существующему `secrets.toml` с `VSELLM_API_KEY`.
3. Запустить `streamlit run streamlit_app.py`.

## Streamlit Community Cloud

1. Создать приложение из этого репозитория, ветка `main`, файл `streamlit_app.py`.
2. В **Advanced settings → Secrets** вставить всё содержимое локального файла
   `SECRETS_FOR_STREAMLIT.toml`.
3. Нажать **Deploy**. Установка `ffmpeg` идёт через `packages.txt`.

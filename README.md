# AI Privacy Assistant

## Структура проекта

- `api/` — Python backend (Flask)
- `public/` — фронтенд (HTML, JS, CSS)
- `credentials/` — приватные ключи (не коммитятся)
- `.env` — переменные окружения
- `requirements.txt` — зависимости Python
- `Procfile` — для Render
- `vercel.json` — для Vercel
- `README.md` — инструкция

## Установка

1. Клонируйте репозиторий
2. Создайте папку `credentials` в корне проекта
3. Поместите файл учетных данных Google Cloud в `credentials/google-cloud.json`
4. Создайте файл `.env` и добавьте необходимые переменные окружения:

```env
GOOGLE_APPLICATION_CREDENTIALS=./credentials/google-cloud.json
GEMINI_API_KEY=ваш_ключ
```

5. Установите зависимости:

```bash
pip install -r requirements.txt
```

6. Для локального запуска:

```bash
python api/index.py
```

7. Для Render: используется Procfile, для Vercel — vercel.json.

**Frontend** лежит в папке `public/`.  
**Backend** — в папке `api/`.  
В production статика отдаётся из `public/`, все API-запросы идут на Flask backend.

## Важно

Для работы вашего проекта установите необходимые зависимости:

pip install flask flask-cors requests PyPDF2 pillow python-docx numpy opencv-python google-cloud-vision

Если используете requirements.txt, добавьте туда:

flask
flask-cors
requests
PyPDF2
pillow
python-docx
numpy
opencv-python
google-cloud-vision

После установки повторно запустите python action.py

**ВАЖНО:**  
Если при загрузке изображения появляется ошибка  
`Google Vision не настроен. Для анонимизации изображений установите переменную окружения GOOGLE_APPLICATION_CREDENTIALS...`  
это значит, что переменная окружения GOOGLE_APPLICATION_CREDENTIALS не установлена.

**Как установить переменную окружения GOOGLE_APPLICATION_CREDENTIALS:**

1. Откройте новое окно командной строки (cmd) или PowerShell.
2. Выполните одну из команд:

   - Для **cmd** (Windows):
     ```
     set GOOGLE_APPLICATION_CREDENTIALS=d:\Савелий\Проекты\Проекты\AI_Hackathon\triple-shift-460105-m7-66f88a86e249.json
     ```
   - Для **PowerShell** (Windows):
     ```
     $env:GOOGLE_APPLICATION_CREDENTIALS="d:\Савелий\Проекты\Проекты\AI_Hackathon\triple-shift-460105-m7-66f88a86e249.json"
     ```
   - Для **Linux/Mac** (bash/zsh):
     ```
     export GOOGLE_APPLICATION_CREDENTIALS="d:/Савелий/Проекты/Проекты/AI_Hackathon/triple-shift-460105-m7-66f88a86e249.json"
     ```

3. В этом же окне запустите сервер:
   ```
   python action.py
   ```

> **Важно:**  
> Переменная окружения должна быть установлена в том же окне, где запускается сервер.  
> Если вы закроете окно — переменная исчезнет, и её нужно будет установить снова.

> **Ограничения Vercel:**  
> На Vercel автоматическое размытие лиц и текста на изображениях не поддерживается из-за ограничений платформы (невозможно установить OpenCV/numpy/dlib).  
> На Vercel блюрится всё изображение целиком. Для точного размытия лиц и текста используйте Render или локальный сервер (см. action.py).

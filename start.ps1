start "MargDrishti Backend" cmd /c "python -m uvicorn backend.app.main:app --reload"
start "MargDrishti Frontend" cmd /c "cd frontend && npm run dev"
echo "Backend starting at http://127.0.0.1:8000"
echo "Frontend starting at http://localhost:5173"

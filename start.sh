#!/bin/bash
python -m uvicorn backend.app.main:app --reload &
cd frontend && npm run dev &
echo "Backend starting at http://127.0.0.1:8000"
echo "Frontend starting at http://localhost:5173"
wait

"""Aggregates every v1 route onto one router."""

from fastapi import APIRouter

from app.api.v1 import ai, analytics, auth, complaints, departments

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(complaints.router)
api_router.include_router(analytics.router)
api_router.include_router(departments.router)
api_router.include_router(ai.router)

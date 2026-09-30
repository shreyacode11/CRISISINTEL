from functools import wraps
from flask import session, redirect, url_for, flash

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login first.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if session.get("role") != "admin":
            flash("Admin access only.", "danger")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


def people_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if session.get("role") != "people":
            flash("People access only.", "danger")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper
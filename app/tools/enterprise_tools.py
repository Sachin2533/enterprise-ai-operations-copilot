from app.database.database import (
    get_enterprise_connection,
    initialize_databases
)


def _ensure_enterprise_database():

    initialize_databases()


def get_employee_info(
    employee_id: str
):
    """
    Get employee details
    """

    _ensure_enterprise_database()

    conn = get_enterprise_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM employees
        WHERE employee_id = ?
        """,
        (employee_id,)
    )

    result = cursor.fetchone()

    conn.close()

    if not result:
        return None

    return {
        "employee_id": result[0],
        "name": result[1],
        "department": result[2]
    }


def get_leave_balance(
    employee_id: str
):
    """
    Get leave balance
    """

    _ensure_enterprise_database()

    conn = get_enterprise_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT casual_leave,
               sick_leave
        FROM leave_balance
        WHERE employee_id = ?
        """,
        (employee_id,)
    )

    result = cursor.fetchone()

    conn.close()

    if not result:
        return None

    return {
        "casual_leave": result[0],
        "sick_leave": result[1]
    }

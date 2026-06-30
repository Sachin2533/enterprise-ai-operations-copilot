from app.database.database import (
    get_memory_connection,
    initialize_memory_database
)


# =====================================
# Conversation Memory
# =====================================

def save_conversation(
    user_message,
    ai_response
):

    initialize_memory_database()

    conn = get_memory_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO conversations
        (
            user_message,
            ai_response
        )
        VALUES (?,?)
        """,
        (
            user_message,
            ai_response
        )
    )

    conn.commit()

    conn.close()


def get_recent_conversations(
    limit=5
):

    initialize_memory_database()

    conn = get_memory_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            user_message,
            ai_response
        FROM conversations
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()

    conn.close()

    return rows[::-1]


# =====================================
# User Profile Memory
# =====================================

def save_employee_id(
    employee_id
):

    initialize_memory_database()

    conn = get_memory_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT OR REPLACE INTO
        user_profiles
        (
            user_key,
            employee_id
        )
        VALUES
        (
            'default_user',
            ?
        )
        """,
        (employee_id,)
    )

    conn.commit()

    conn.close()


def get_employee_id():

    initialize_memory_database()

    conn = get_memory_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT employee_id
        FROM user_profiles
        WHERE user_key='default_user'
        """
    )

    result = cursor.fetchone()

    conn.close()

    if result:
        return result[0]

    return None

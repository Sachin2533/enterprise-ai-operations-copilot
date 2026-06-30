import re
import operator


OPERATIONS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv
}


def calculate(expression: str):
    """
    Supports:
    10 + 5
    20 * 3
    100 / 4
    """

    expression = expression.strip()

    pattern = r"^\s*(-?\d+(?:\.\d+)?)\s*([\+\-\*/])\s*(-?\d+(?:\.\d+)?)\s*$"

    match = re.match(pattern, expression)

    if not match:
        return "Invalid expression. Example: 10 + 5"

    left = float(match.group(1))
    op = match.group(2)
    right = float(match.group(3))

    try:

        result = OPERATIONS[op](
            left,
            right
        )

        if result == int(result):
            return str(int(result))

        return str(round(result, 4))

    except ZeroDivisionError:
        return "Cannot divide by zero."
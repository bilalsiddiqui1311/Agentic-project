import ast
import operator
import re
from collections.abc import Callable

from app.tools.base import ToolOutput


class CalculatorTool:
    name = "calculator"
    description = "Safely evaluates simple arithmetic expressions."

    _operators: dict[type[ast.operator] | type[ast.unaryop], Callable[..., float]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def run(self, query: str) -> ToolOutput:
        expression = self._extract_expression(query)
        value = self._evaluate(expression)
        return ToolOutput(name=self.name, content=f"{expression} = {self._format(value)}")

    def _extract_expression(self, query: str) -> str:
        cleaned = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", query.lower())
        expression = "".join(char for char in cleaned if char in "0123456789.+-*/()% ")
        expression = re.sub(r"\s+", " ", expression).strip()

        if not expression or not any(char.isdigit() for char in expression):
            raise ValueError("I could not find an arithmetic expression to calculate.")

        if len(expression) > 100:
            raise ValueError("That expression is too long for this starter calculator.")

        return expression

    def _evaluate(self, expression: str) -> float:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError("That arithmetic expression is not valid.") from exc

        return self._eval_node(tree.body)

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value

        if isinstance(node, ast.BinOp):
            operator_fn = self._operators.get(type(node.op))
            if operator_fn is None:
                raise ValueError("That operator is not supported.")

            left = self._eval_node(node.left)
            right = self._eval_node(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 8:
                raise ValueError("Exponents above 8 are blocked in this starter calculator.")

            try:
                return operator_fn(left, right)
            except ZeroDivisionError as exc:
                raise ValueError("Division by zero is not allowed.") from exc

        if isinstance(node, ast.UnaryOp):
            operator_fn = self._operators.get(type(node.op))
            if operator_fn is None:
                raise ValueError("That unary operator is not supported.")
            return operator_fn(self._eval_node(node.operand))

        raise ValueError("Only simple numbers and arithmetic operators are supported.")

    def _format(self, value: float) -> str:
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

import math

class Math:
    @staticmethod
    def add(a: float, b: float) -> float:
        """Add two numbers and returns the sum"""
        return a + b

    @staticmethod
    def subtract(a: float, b: float) -> float:
        """Subtract two numbers and returns the difference"""
        return a - b

    @staticmethod
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers and returns the product"""
        return a * b

    @staticmethod
    def divide(a: float, b: float) -> float:
        """Divide two numbers and returns the quotient"""
        return a / b


class PaintCostCalculator:

    @staticmethod
    def calculate_paint_cost(area: float, price_per_gallon: float, add_paint_supply_costs: bool = False) -> float:
        """
        Calculate the total cost of paint needed for a given area.
        
        Args:
            area: Area to be painted in square feet
            price_per_gallon: Price per gallon of paint
            add_paint_supply_costs: Whether to add $50 for painting supplies
            
        Returns:
            Total cost of paint and supplies if requested
        """
        gallons_needed = math.ceil((area / 400) * 2) # Assuming 2 gallons are needed for 400 square feet
        total_cost = round(gallons_needed * price_per_gallon, 2)
        if add_paint_supply_costs:
            total_cost += 50
        return total_cost

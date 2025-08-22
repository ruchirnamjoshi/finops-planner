# 🤝 Contributing to FinOps Planner

Thank you for your interest in contributing to FinOps Planner! We welcome contributions from the community and appreciate your help in making this project better.

## 🚀 **Getting Started**

### **Prerequisites**
- Python 3.8+
- Git
- OpenAI API key (for testing)
- Basic understanding of FinOps concepts

### **Development Setup**
1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/finops-planner.git
   cd finops-planner
   ```
3. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
5. **Set up environment variables**
   ```bash
   cp env_template.txt .env
   # Add your OpenAI API key to .env
   ```

## 📋 **Contribution Guidelines**

### **What We're Looking For**
- **Bug fixes** and improvements
- **New features** that align with our roadmap
- **Documentation** improvements
- **Test coverage** enhancements
- **Performance optimizations**
- **UI/UX improvements**

### **What to Avoid**
- **Breaking changes** without discussion
- **Large refactoring** without prior approval
- **Adding dependencies** without justification
- **Style-only changes** (use automated tools instead)

## 🔧 **Development Workflow**

### **1. Create a Feature Branch**
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-description
```

### **2. Make Your Changes**
- Follow our coding standards
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### **3. Test Your Changes**
```bash
# Run the application
streamlit run app.py

# Run tests (if available)
python -m pytest tests/

# Check code quality
python -m flake8 planner/
python -m black --check planner/
```

### **4. Commit Your Changes**
```bash
git add .
git commit -m "feat: add new optimization strategy

- Added new cost optimization algorithm
- Updated documentation
- Added unit tests"
```

### **5. Push and Create a Pull Request**
```bash
git push origin feature/your-feature-name
```

## 📝 **Coding Standards**

### **Python Style Guide**
- Follow **PEP 8** guidelines
- Use **type hints** for all function parameters and return values
- Keep functions under **50 lines** when possible
- Use **descriptive variable names**

### **Code Example**
```python
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class OptimizationStrategy(BaseModel):
    """Represents a cost optimization strategy."""
    
    name: str = Field(..., description="Strategy name")
    savings_potential: float = Field(..., description="Expected savings percentage")
    implementation_effort: str = Field(..., description="Effort level (low/medium/high)")
    
    def calculate_savings(self, current_cost: float) -> float:
        """Calculate actual savings amount."""
        return current_cost * (self.savings_potential / 100)
```

### **Documentation Standards**
- **Docstrings** for all public functions and classes
- **README updates** for new features
- **Inline comments** for complex logic
- **Type hints** for better code understanding

### **Testing Requirements**
- **Unit tests** for new functionality
- **Integration tests** for agent interactions
- **Test coverage** should not decrease
- **Mock external dependencies** (OpenAI API, etc.)

## 🧪 **Testing Guidelines**

### **Running Tests**
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_optimizer.py

# Run with coverage
python -m pytest --cov=planner

# Run with verbose output
python -m pytest -v
```

### **Writing Tests**
```python
import pytest
from planner.optimizer_bot import IntelligentCostOptimizerAgent

class TestOptimizerAgent:
    """Test cases for the optimizer agent."""
    
    def test_optimization_generation(self):
        """Test that optimization recommendations are generated correctly."""
        agent = IntelligentCostOptimizerAgent()
        # Add your test logic here
        assert True  # Replace with actual assertions
```

## 📚 **Documentation**

### **What to Document**
- **New features** and their usage
- **API changes** and breaking changes
- **Configuration options** and environment variables
- **Deployment instructions** for new components

### **Documentation Format**
- Use **Markdown** for all documentation
- Include **code examples** where appropriate
- Add **screenshots** for UI changes
- Update **README.md** for major changes

## 🐛 **Bug Reports**

### **Before Reporting**
1. **Check existing issues** to avoid duplicates
2. **Search documentation** for solutions
3. **Test with latest version** from main branch

### **Bug Report Template**
```markdown
## Bug Description
Brief description of the issue

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., macOS 14.0]
- Python: [e.g., 3.9.0]
- FinOps Planner: [e.g., commit hash]

## Additional Information
Screenshots, logs, or other relevant information
```

## 💡 **Feature Requests**

### **Feature Request Template**
```markdown
## Feature Description
Brief description of the requested feature

## Use Case
Why this feature is needed and how it would be used

## Proposed Solution
Your suggested approach to implementing this feature

## Alternatives Considered
Other approaches you've considered

## Additional Information
Any other relevant details
```

## 🔄 **Pull Request Process**

### **Before Submitting**
1. **Ensure tests pass** locally
2. **Update documentation** as needed
3. **Follow coding standards**
4. **Squash commits** if there are many small changes

### **Pull Request Template**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Documentation updated

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] No breaking changes introduced
```

## 🏷️ **Issue Labels**

We use the following labels to categorize issues:
- **bug**: Something isn't working
- **enhancement**: New feature or request
- **documentation**: Improvements or additions to documentation
- **good first issue**: Good for newcomers
- **help wanted**: Extra attention is needed
- **priority: high**: Important issue
- **priority: low**: Low priority issue

## 📞 **Getting Help**

### **Community Support**
- **GitHub Discussions**: Ask questions and share ideas
- **GitHub Issues**: Report bugs and request features
- **Code Reviews**: Get feedback on your contributions

### **Contact the Team**
- **Email**: support@finops-planner.com
- **GitHub**: @ruchirnamjoshi

## 🙏 **Recognition**

Contributors will be recognized in:
- **README.md** contributors section
- **Release notes** for significant contributions
- **GitHub contributors** page

---

**Thank you for contributing to FinOps Planner! 🚀**

Your contributions help make cloud cost optimization more intelligent and accessible for everyone.

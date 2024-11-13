# 방법 1: graphviz 사용
from graphviz import Digraph

def create_pipeline_flowchart_graphviz(output_file='pipeline_flow'):
    """
    Create a flowchart using graphviz
    
    Args:
        output_file: 출력 파일명 (확장자 제외)
    """
    dot = Digraph(comment='Property Data Pipeline')
    dot.attr(rankdir='TB')  # Top to Bottom 방향
    
    # Add nodes
    dot.node('A', 'Raw Data')
    dot.node('B', 'Feature Type Check')
    
    # Feature type nodes
    dot.node('C1', 'Categorical Features')
    dot.node('C2', 'Coordinate/Date Features')
    dot.node('C3', 'High Skewness Features')
    dot.node('C4', 'Other Features')
    
    # Scaler nodes
    dot.node('D1', 'StandardScaler')
    dot.node('D2', 'RobustScaler')
    dot.node('D3', 'PowerTransformer')
    dot.node('D4', 'RobustScaler')
    
    # Process nodes
    dot.node('E', 'Scaled Features')
    dot.node('F', 'Correlation Analysis')
    dot.node('G', 'Feature Importance')
    dot.node('H', 'Group Correlated Features')
    dot.node('I', 'Calculate Importance Scores')
    dot.node('J', 'Feature Selection')
    dot.node('K', 'Select Best Feature from Each Group')
    dot.node('L', 'Keep Important Uncorrelated Features')
    dot.node('M', 'Final Feature Set')
    
    # Add edges
    dot.edge('A', 'B')
    dot.edge('B', 'C1')
    dot.edge('B', 'C2')
    dot.edge('B', 'C3')
    dot.edge('B', 'C4')
    
    dot.edge('C1', 'D1')
    dot.edge('C2', 'D2')
    dot.edge('C3', 'D3')
    dot.edge('C4', 'D4')
    
    for d in ['D1', 'D2', 'D3', 'D4']:
        dot.edge(d, 'E')
    
    dot.edge('E', 'F')
    dot.edge('E', 'G')
    dot.edge('F', 'H')
    dot.edge('G', 'I')
    dot.edge('H', 'J')
    dot.edge('I', 'J')
    dot.edge('J', 'K')
    dot.edge('J', 'L')
    dot.edge('K', 'M')
    dot.edge('L', 'M')
    
    # Save the flowchart
    dot.render(output_file, view=True, format='png')


# 방법 2: mermaid-py 사용
# from mermaid import Mermaid

# # def create_pipeline_flowchart_mermaid(output_file='pipeline_flow.md'):
# #     """
# #     Create a flowchart using mermaid-py
    
# #     Args:
# #         output_file: 출력 파일명 (.md 확장자 권장)
# #     """
# #     m = Mermaid()
    
# #     # Define the flowchart
# #     flowchart = """
# #     flowchart TD
# #         A[Raw Data] --> B[Feature Type Check]
# #         B --> C1[Categorical Features]
# #         B --> C2[Coordinate/Date Features]
# #         B --> C3[High Skewness Features]
# #         B --> C4[Other Features]
        
# #         C1 --> D1[StandardScaler]
# #         C2 --> D2[RobustScaler]
# #         C3 --> D3[PowerTransformer]
# #         C4 --> D4[RobustScaler]
        
# #         D1 & D2 & D3 & D4 --> E[Scaled Features]
        
# #         E --> F[Correlation Analysis]
# #         E --> G[Feature Importance]
        
# #         F --> H[Group Correlated Features]
# #         G --> I[Calculate Importance Scores]
        
# #         H & I --> J[Feature Selection]
# #         J --> K[Select Best Feature from Each Group]
# #         J --> L[Keep Important Uncorrelated Features]
        
# #         K & L --> M[Final Feature Set]
#     """
    
#     # Create and save the flowchart
#     m.content = flowchart
#     with open(output_file, 'w') as f:
#         f.write(str(m))

# 사용 예시
if __name__ == "__main__":
    # Graphviz 방식
    create_pipeline_flowchart_graphviz()
    
    # # Mermaid 방식
    # create_pipeline_flowchart_mermaid()
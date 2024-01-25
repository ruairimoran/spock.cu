#include "src/tree.cu"


int main(void)
{
    std::ifstream file("scenario-tree.json"); 
    ScenarioTree tree(file);
  	tree.print();	
}

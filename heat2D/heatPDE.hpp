#ifndef HEATPDE_HPP
#define HEATPDE_HPP

namespace heat_pde
{

class HeatPDE
{
    private:
    public:
        HeatPDE(const HeatPDE&) = delete;
        HeatPDE& operator=(const HeatPDE&) = delete;
        HeatPDE(const HeatPDE&&) = delete;
        HeatPDE& operator=(const HeatPDE&&) = delete;
        
        // TODO define specialised constructor:
        HeatPDE();
};

} // namespace
#endif // ifndef HEATPDE_HPP
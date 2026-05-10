import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { VisionAssistant } from "@/components/vision-assistant";
import { Simulator } from "@/pages/simulator";
import { useEffect } from "react";
import { Route, Switch } from "wouter";

const queryClient = new QueryClient();

function App() {
  useEffect(() => {
    document.documentElement.classList.add("dark");
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Switch>
          <Route path="/simulator" component={Simulator} />
          <Route component={VisionAssistant} />
        </Switch>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;

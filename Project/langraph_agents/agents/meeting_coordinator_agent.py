# TODO: Create MeetingCoordinatorAgent class inheriting from BaseAgent
# TODO: Implement generate_agenda creating meeting agenda items
# TODO: Implement generate_talking_points from analysis results
# TODO: Implement generate_questions creating relevant questions for prospect
# TODO: Implement generate_objection_handling creating responses to common objections
# TODO: Implement generate_next_steps creating follow-up action items
# TODO: Implement async run method:
#   - Review all analysis results (risk, persona, products, recommendations)
#   - Generate meeting agenda
#   - Create talking points highlighting key findings
#   - Generate questions to ask prospect
#   - Create objection handling strategies
#   - Generate next steps for follow-up
#   - Return MeetingGuide in state

from typing import Dict,Any,List
from langchain_core.prompts import ChatPromptTemplate


from langraph_agents.base_agent import BaseAgent
from state import WorkflowState, MeetingGuide
from config.settings import get_settings

class MeetingCoordinatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name = "Meeting Coordinator Agent",
            description = "Generates meeting guides, agendas, reports and preparations materials for client meetings."
        )
        self.settings = get_settings() #Come back to this, hasn't been implemented yet

    async def execute(self, state: WorkflowState)->WorkflowState:
        
        self.logger.info("Generating meeting guides")

        prospect_data = state.prospect.prospect_data
        risk_assessment = state.analysis.risk_assessment
        persona_classification = state.persona_classification
        recommendations = state.recommendations.recommended_products #Maybe cause issues in state.py


        if not prospect_data:
            raise ValueError("Prospect Data not available for guide generation")
        
        meeting_guide = await self._generate_meeting_guide(
            prospect_data, risk_assessment, persona_classification, recommendations
        )

        presentation_slides = await self._generate_presentation_outline(
            prospect_data, risk_assessment, persona_classification, recommendations
        )

        client_materials = self._generate_client_materials_list(
            prospect_data, risk_assessment, recommendations
        )

        state.meeting.meeting_guide = meeting_guide
        state.meeting.presentation_slides = presentation_slides
        state.meeting.client_materials = client_materials

        self.logger.info("Meeting guide generated!")
        return state

    async def _generate_meeting_guide(self, prospect_data, risk_assessment, persona_classification, recommendations)->MeetingGuide:

        agenda_items = await self._generate_agenda(prospect_data,risk_assessment,persona_classification)
        talking_points = await self._generate_talking_points(prospect_data, risk_assessment, recommendations)
        questions = await self._generate_questions(prospect_data, persona_classification)
        next_steps = await self._generate_next_steps(prospect_data, recommendations)
        objection_handling=await self._generate_objection_handling(risk_assessment,persona_classification)


        duration = self._estimate_meeting_duration(agenda_items,talking_points,questions)


        return MeetingGuide(
            agenda_items = agenda_items,
            key_talking_points = talking_points,
            questions_to_ask = questions,
            objection_handling = objection_handling,
            next_steps = next_steps,
            estimated_duration = duration
        )

    async def _generate_agenda(self,prospect_data,risk_assessment, persona_classification) -> List[str]:
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system",self.get_system_prompt()),
            ("human","""
            Generate a structured meeting agenda for this client consultation:

            Client: {client_name}
            Age: {age}
            Risk Profile: {risk_level}
            Persona: {persona_type}
            Investment Goal: {investment_goal}

            Create 6-7 agenda items that cover the following:
            1.Warm welcome and relationship building
            2.Understanding client needs
            3.Risk assessment discussion
            4.Product presentation
            5.Next steps and follow-up

            Give the output formatted as a bullet list with estimated time for each items.
            """)

        ])

        input_variables = {
            "client_name": prospect_data.name,
            "age":prospect_data.age,
            "risk_level": risk_assessment.risk_level if risk_assessment else "To be declared",
            "persona_type": persona_classification.persona_type if persona_classification else "To be declared",
            "investment_goal": prospect_data.investment_goal or "General investment planning"
        }

        response = await self.generate_response(prompt_template, input_variables)

        return self._parse_bulleted_list(response)

    async def _generate_talking_points(self, prospect_data, risk_assessment,recommendations) -> List[str]:

        prompt_template = ChatPromptTemplate.from_messages([
            ("system",self.get_system_prompt()),
            ("human","""
            Generate key talking points for discussing with this client

            Client: {client_name}
            Age: {age}
            Income: Rupees{annual_income:,}
            Target Goal: Rupees{target_goal:,}
            Horizon: {investment_horizon} years

            Risk Assessment: {risk_level}

            Top Recommendations: {recommendations_summary}
    

            Create talking points that:
            1.Build rapport and trust
            2.Demonstrate understanding of the clients' needs
            3.Present solutions effectively
            4.Address potential concerns
            5.Create urgency and next

            Give the output formatted as clear, actionable talking points
            """)

        ])

        recommendations_summary = ""
        if recommendations:
            recommendations_summary = "\n".join([
                f"- {rec.product_name}: {rec.justification[:100]}..."
                for rec in recommendations[:3]
            ])

        input_variables = {
            "client_name": prospect_data.name,
            "age":prospect_data.age,
            "annual_income": prospect_data.annual_income,
            "target_goal":prospect_data.target_goal_amount,
            "investment_horizon":prospect_data.investment_horizon_years,
            "risk_level": risk_assessment.risk_level if risk_assessment else "To be assessed",
            "recommendedations_summary": recommendations_summary or "To be presented"
        }

        response = await self.generate_response(prompt_template, input_variables)

        return self._parse_bulleted_list(response)    

    async def _generate_questions(self, prospect_data, persona_classification)-> List[str]:

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("human","""

            Generate discovery questions to ask this client during the meeting:

            Client: {client_name}
            Age: {age}
            Persona: {persona_type}
            Investment Goal: {investment_goal}

            Create questions that:
            1. Uncover deeper financial goals, ambitions and motivations
            2. Assess risk tolerance and investment experience
            3. Understand family and lifestyle factors
            4. Identify potential objections or concerns
            5. Gauge decision-making process and timeline

            Focus on open-ended questions that encourage dialouge. 
            Format it as a list of questions
            
            
            """)
        ])

        input_variables = {
            "client_name": prospect_data.name,
            "age": prospect_data.age,
            "persona_type": persona_classification.persona_type if persona_classification else "To be determined",
            "investment_goal": prospect_data.investment_goal or "General investment planning"
        }
        
        response = await self.generate_response(prompt_template, input_variables)
        return self._parse_questions(response)
    
    async def _generate_objection_handling(self, risk_assessment, persona_classification) -> Dict[str, str]:
        """Generate objection handling strategies."""
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("human", """
            Generate objection handling strategies for this client profile:
            
            Risk Profile: {risk_level}
            Persona: {persona_type}
            
            Create responses for common objections:
            1. "The fees seem high"
            2. "I need to think about it"
            3. "I'm not comfortable with this risk level"
            4. "I want to compare with other options"
            5. "I don't have enough money to invest"
            
            Format as: Objection -> Response strategy
            Keep responses professional and empathetic.
            """)
        ])
        
        input_variables = {
            "risk_level": risk_assessment.risk_level if risk_assessment else "To be determined",
            "persona_type": persona_classification.persona_type if persona_classification else "To be determined"
        }
        
        response = await self.generate_response(prompt_template, input_variables)
        return self._parse_objection_responses(response)
    
    async def _generate_next_steps(self, prospect_data, recommendations) -> List[str]:
        """Generate next steps and follow-up actions."""
        next_steps = [
            "Review and finalize investment recommendations",
            "Complete KYC and documentation process",
            "Set up investment accounts if needed",
            "Schedule follow-up meeting in 2 weeks",
            "Provide additional educational materials"
        ]
        

        if prospect_data.investment_experience_level == "Beginner":
            next_steps.insert(1, "Provide investor education materials")
        
        if recommendations and len(recommendations) > 3:
            next_steps.insert(2, "Prioritize top 3 investment options")
        
        return next_steps
    
    def _estimate_meeting_duration(self, agenda_items: List[str], talking_points: List[str], questions: List[str]) -> int:

        base_duration = 45  # Base meeting time
        

        agenda_time = len(agenda_items) * 5
        talking_points_time = len(talking_points) * 3
        questions_time = len(questions) * 2
        
        total_duration = base_duration + agenda_time + talking_points_time + questions_time
        

        return ((total_duration + 14) // 15) * 15
    
    async def _generate_presentation_outline(
        self, 
        prospect_data, 
        risk_assessment, 
        persona_classification, 
        recommendations
    ) -> List[str]:

        slides = [
            "Welcome & Agenda",
            f"Understanding {prospect_data.name}'s Goals",
            "Risk Assessment & Profile",
            "Investment Strategy Overview",
            "Recommended Solutions",
            "Implementation Timeline",
            "Next Steps & Questions"
        ]
        

        if persona_classification:
            if persona_classification.persona_type == "Aggressive Growth":
                slides.insert(-2, "Growth Opportunities & Market Outlook")
            elif persona_classification.persona_type == "Cautious Planner":
                slides.insert(-2, "Risk Management & Capital Protection")
        
        return slides
    
    def _generate_client_materials_list(self, prospect_data, risk_assessment, recommendations) -> List[str]:

        materials = [
            "Client profile summary",
            "Risk assessment questionnaire",
            "Product fact sheets",
            "Fee structure document",
            "Investment application forms"
        ]
        

        if recommendations:
            for rec in recommendations[:3]:
                materials.append(f"{rec.product_name} - detailed brochure")
        

        materials.extend([
            "Risk disclosure statements",
            "Terms and conditions",
            "Privacy policy"
        ])
        
        return materials
    
    def _parse_bulleted_list(self, text: str) -> List[str]:

        items = []
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('•', '-', '*', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):

                clean_line = line.lstrip('•-*123456789. ').strip()
                if clean_line:
                    items.append(clean_line)
        return items or ["Meeting agenda to be customized"]
    
    def _parse_questions(self, text: str) -> List[str]:

        questions = []
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if '?' in line:
                # Extract the question
                if line.startswith(('•', '-', '*')):
                    line = line.lstrip('•-* ').strip()
                questions.append(line)
        return questions or ["What are your primary investment objectives?"]
    
    def _parse_objection_responses(self, text: str) -> Dict[str, str]:

        objections = {}
        lines = text.split('\n')
        current_objection = None
        
        for line in lines:
            line = line.strip()
            if '->' in line or ':' in line:
                parts = line.split('->' if '->' in line else ':', 1)
                if len(parts) == 2:
                    objection = parts[0].strip().strip('"')
                    response = parts[1].strip()
                    objections[objection] = response
        

        if not objections:
            objections = {
                "Fees seem high": "Let me show you the value proposition and long-term benefits",
                "Need to think about it": "I understand. What specific concerns can I address today?",
                "Risk level concerns": "Let's discuss risk management strategies that align with your comfort level"
            }
        
        return objections
    
    def get_prompt_template(self) -> ChatPromptTemplate:

        return ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("human", "Generate meeting preparation materials for the given client profile.")
        ])
    
    def validate_input(self, state: WorkflowState) -> bool:

        return state.prospect.prospect_data is not None
    
    def validate_output(self, state: WorkflowState) -> bool:

        return (
            state.meeting.meeting_guide is not None and
            len(state.meeting.meeting_guide.agenda_items) > 0
        )
    

import streamlit as st
import requests
import json

BASE_URL = "http://localhost:8000/api"

st.set_page_config(page_title="SLM Gateway UI", layout="wide")

st.title("SLM Recruiting Gateway")

if "job_spec" not in st.session_state:
    st.session_state.job_spec = None
if "candidate_profile" not in st.session_state:
    st.session_state.candidate_profile = None

tab1, tab2, tab3 = st.tabs(["1. AI Job Creation", "2. Add Candidate", "3. Match"])

# -----------------------------------------------------------------------------
# Tab 1: AI Job Creation
# -----------------------------------------------------------------------------
with tab1:
    st.header("Create a Job")
    with st.form("job_form"):
        title = st.text_input("Job Title *")
        skills = st.text_input("Skills (comma separated)")
        prompt = st.text_area("Detailed Prompt / Job Requirements (Optional)", help="Provide a detailed prompt for the SLM to generate a proper job description.")
        min_years = st.number_input("Minimum Years of Experience", min_value=0.0, step=0.5, value=0.0)
        seniority = st.selectbox("Seniority", ["", "Junior", "Mid", "Senior", "Lead", "Principal"])
        enrich = st.checkbox("Use AI to enrich job description", value=True)
        
        submit_job = st.form_submit_button("Generate Job Spec")
        
    if submit_job:
        if not title:
            st.error("Job title is required.")
        else:
            with st.spinner("Generating job specification..."):
                skill_list = [s.strip() for s in skills.split(",") if s.strip()]
                payload = {
                    "title": title,
                    "skills": skill_list,
                    "prompt": prompt if prompt else None,
                    "min_years_experience": min_years if min_years > 0 else None,
                    "seniority": seniority if seniority else None,
                    "enrich": enrich
                }
                try:
                    resp = requests.post(f"{BASE_URL}/jobs", json=payload)
                    if resp.status_code == 200:
                        st.session_state.job_spec = resp.json()
                        st.success("Job generated successfully!")
                    else:
                        st.error(f"Error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

    if st.session_state.job_spec:
        st.markdown("---")
        st.subheader("Current Job Spec")
        job = st.session_state.job_spec
        st.write(f"**Title:** {job.get('title')}")
        st.write(f"**Seniority:** {job.get('seniority')}")
        st.write(f"**Required Skills:** {', '.join(job.get('required_skills', []))}")
        st.write(f"**Preferred Skills:** {', '.join(job.get('preferred_skills', []))}")
        with st.expander("Responsibilities"):
            for r in job.get('responsibilities', []):
                st.markdown(f"- {r}")
        with st.expander("Qualifications"):
            for q in job.get('qualifications', []):
                st.markdown(f"- {q}")
        with st.expander("Full Description"):
            st.markdown(job.get('description', ''))


# -----------------------------------------------------------------------------
# Tab 2: Add Candidate
# -----------------------------------------------------------------------------
with tab2:
    st.header("Upload Candidate Resume")
    uploaded_file = st.file_uploader("Upload PDF Resume", type=["pdf", "png", "jpg", "jpeg"])
    if st.button("Parse Resume") and uploaded_file:
        with st.spinner("Parsing and extracting..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                resp = requests.post(f"{BASE_URL}/resume/parse", files=files)
                if resp.status_code == 200:
                    st.session_state.candidate_profile = resp.json()
                    st.session_state.candidate_summary = None  # reset old summary
                    st.success("Resume parsed successfully!")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")
                
    if st.session_state.candidate_profile:
        st.markdown("---")
        st.subheader("Parsed Profile")
        cand = st.session_state.candidate_profile
        contact = cand.get('contact', {})
        st.write(f"**Name:** {contact.get('name')}")
        st.write(f"**Email:** {contact.get('email')}")
        st.write(f"**Phone:** {contact.get('phone')}")
        
        if cand.get('summary'):
            st.info(cand.get('summary'))
            
        st.write(f"**Total Experience:** {cand.get('total_years_experience')} years")
        st.write(f"**Skills:** {', '.join(cand.get('skills', []))}")
        
        with st.expander("Experience"):
            for exp in cand.get('experience', []):
                st.markdown(f"**{exp.get('title')}** at {exp.get('company')} ({exp.get('start')} - {exp.get('end')})")
                for h in exp.get('highlights', []):
                    st.markdown(f"- {h}")
        
        with st.expander("Education"):
            for edu in cand.get('education', []):
                st.markdown(f"**{edu.get('degree')} in {edu.get('field')}** from {edu.get('institution')} ({edu.get('year')})")

        # ---- AI Summary (on-demand, comprehensive) ----
        st.markdown("---")
        if st.button("🧠 Generate AI Summary", type="primary"):
            with st.spinner("Analyzing the full resume and computing per-skill experience..."):
                try:
                    resp = requests.post(f"{BASE_URL}/candidate/summary", json=cand, timeout=900)
                    if resp.status_code == 200:
                        st.session_state.candidate_summary = resp.json()
                    else:
                        st.error(f"Error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

        summ = st.session_state.get("candidate_summary")
        if summ:
            st.subheader("🧠 AI Candidate Summary")
            if summ.get("total_years_experience") is not None:
                st.metric("Total Experience", f"{summ['total_years_experience']:g} years")
            st.markdown(summ.get("summary", ""))

            skill_exp = summ.get("skill_experience", [])
            evidenced = [s for s in skill_exp if s.get("evidenced")]
            if evidenced:
                st.markdown("**Years of experience per skill:**")
                st.dataframe(
                    [{"Skill": s["skill"], "Years": s["years"]} for s in evidenced],
                    use_container_width=True, hide_index=True,
                )
            other = [s["skill"] for s in skill_exp if not s.get("evidenced")]
            if other:
                st.caption("Also listed (not tied to a dated role): " + ", ".join(other))

            if summ.get("strengths"):
                st.markdown("**Strong points:**")
                for s in summ["strengths"]:
                    st.markdown(f"- {s}")


# -----------------------------------------------------------------------------
# Tab 3: Match
# -----------------------------------------------------------------------------
with tab3:
    st.header("Match Candidate to Job")
    
    if st.session_state.job_spec and st.session_state.candidate_profile:
        st.write("Using the Job and Candidate from your session.")
        if st.button("Calculate Match Score"):
            with st.spinner("Matching..."):
                payload = {
                    "job": st.session_state.job_spec,
                    "candidate": st.session_state.candidate_profile,
                    "justify": True
                }
                try:
                    resp = requests.post(f"{BASE_URL}/match", json=payload)
                    if resp.status_code == 200:
                        match_res = resp.json()
                        score = match_res.get('overall_score', 0)
                        
                        st.metric("Match Score", f"{score:.1f} / 100", match_res.get('verdict'))
                        st.progress(score / 100.0)
                        
                        if match_res.get('recommendation'):
                            st.markdown(f"**AI Recommendation:** {match_res.get('recommendation')}")
                        if match_res.get('justification'):
                            st.markdown(f"**Justification:**\n\n{match_res.get('justification')}")
                        
                        st.write("### Skills Analysis")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("✅ **Matched Skills:**")
                            for s in match_res.get('matched_skills', []):
                                st.write(f"- **{s.get('skill')}** *(Evidence: {s.get('evidence')})*")
                        with col2:
                            st.write("❌ **Missing Skills:**")
                            for s in match_res.get('missing_skills', []):
                                st.write(f"- {s}")
                    else:
                        st.error(f"Error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")
    else:
        st.warning("Please create a job and add a candidate in the previous tabs first.")
        
        st.markdown("---")
        st.write("Or use the one-shot match endpoint (direct upload):")
        with st.form("oneshot_form"):
            os_title = st.text_input("Job Title *")
            os_skills = st.text_input("Skills (comma separated)")
            os_prompt = st.text_area("Detailed Prompt / Job Requirements (Optional)")
            os_file = st.file_uploader("Upload Resume *", type=["pdf", "png", "jpg"])
            os_submit = st.form_submit_button("Match")
            
        if os_submit:
            if not os_title or not os_file:
                st.error("Title and Resume are required.")
            else:
                with st.spinner("Matching..."):
                    files = {"file": (os_file.name, os_file.getvalue(), os_file.type)}
                    data = {
                        "title": os_title,
                        "skills": os_skills,
                        "prompt": os_prompt if os_prompt else None
                    }
                    try:
                        resp = requests.post(f"{BASE_URL}/match/upload", files=files, data=data)
                        if resp.status_code == 200:
                            match_res = resp.json()
                            score = match_res.get('overall_score', 0)
                            
                            st.metric("Match Score", f"{score:.1f} / 100", match_res.get('verdict'))
                            st.progress(score / 100.0)
                            
                            if match_res.get('recommendation'):
                                st.markdown(f"**AI Recommendation:** {match_res.get('recommendation')}")
                            if match_res.get('justification'):
                                st.markdown(f"**Justification:**\n\n{match_res.get('justification')}")
                                
                            st.write("### Skills Analysis")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("✅ **Matched Skills:**")
                                for s in match_res.get('matched_skills', []):
                                    st.write(f"- **{s.get('skill')}** *(Evidence: {s.get('evidence')})*")
                            with col2:
                                st.write("❌ **Missing Skills:**")
                                for s in match_res.get('missing_skills', []):
                                    st.write(f"- {s}")
                        else:
                            st.error(f"Error {resp.status_code}: {resp.text}")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")
